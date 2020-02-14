import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style
import torch.nn as nn
import models.classifier
from models import mnn
from models.layerbuilder import LayerMetaData
import config
from datasets import package
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import numpy as np
import torch.nn.functional as F
from data_augments import TpsAndRotate, TpsAndRotateSecond
from utils.viewer import UniImageViewer
from apex import amp
import wandb
import it
from utils.text import text_patch
import pygame

global_step = 0.0


class Guesser():
    def __init__(self, n):
        self.hist = torch.zeros(n, n)

    def add(self, y, target):
        """

        :param y: a distribution over classes(C), dimensions B x C
        :param target: the known correct target labels, dimension B
        :return:
        """

        y = torch.argmax(y, dim=1)
        self.hist[target, y] += 1
        guess = torch.argmax(self.hist, dim=0)
        return guess[y]

    def guess(self):
        return torch.argmax(self.hist, dim=0)


def show(x, y, columns=10):
    empty = torch.zeros_like(x[0])
    clazz = torch.argmax(y, dim=1)
    assignment = [[] for _ in range(y.size(1))]
    for image, cls in zip(x, clazz):
        if len(assignment[cls]) < columns:
            assignment[cls].append(image)

    for row in assignment:
        row += [empty for _ in range(columns - len(row))]

    rows = []
    for column in assignment:
        rows.append(torch.cat(column, dim=2))
    return torch.cat(rows, dim=1)


def mi(P):
    eps = torch.finfo(P.dtype).eps
    P[P < eps] = eps
    m0 = torch.sum(P, dim=0, keepdim=True)
    m1 = torch.sum(P, dim=1, keepdim=True)
    return torch.sum((P.log2() - m0.log2() - m1.log2()) * P)


def main(args):
    def precision(confusion, totals=False):
        correct = confusion * torch.eye(confusion.shape[0])
        incorrect = confusion - correct
        correct = correct.sum(0)
        incorrect = incorrect.sum(0)
        precision = correct / (correct + incorrect)
        total_correct = correct.sum().item()
        total_incorrect = incorrect.sum().item()
        percent_correct = total_correct / (total_correct + total_incorrect)
        if totals:
            return precision, percent_correct, total_correct, total_correct + total_incorrect
        return precision, percent_correct

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    class Batch:
        def __init__(self, type, loader, dataset):
            self.type = type
            self.loader = loader
            self.batch = tqdm(loader, total=len(dataset) // args.batchsize)
            self.batches = len(dataset) // args.batchsize
            self.len_dataset = len(dataset)
            self.ll = 0
            self.confusion = torch.zeros(datapack.num_classes, datapack.num_classes)
            self.total = 0
            self.correct = 0
            self.batch_step = 0
            self.guesser = Guesser(10)
            self.viewer = UniImageViewer(args.dataset_name, screen_resolution=(1480, 1280))
            self.total_n = 0
            self.classes = datapack.class_list

        def __iter__(self):
            return iter(self.batch)

        def log_step(self):
            global global_step
            self.batch_step += 1
            self.ll += loss.detach().item()

            predicted = self.guesser.add(y, target)
            for p, t in zip(predicted, target):
                self.confusion[p, t] += 1

            guesses = self.guesser.guess()
            label_text = []
            for guess in guesses:
                class_txt = self.classes[guess]
                correct = self.confusion[guess, guess]
                total = self.confusion[guess].sum()
                txt = f'  {class_txt}   {correct} / {total}'
                label_text.append(text_patch(txt, (x.shape[1], x.shape[2], 200), fontsize=20))
            label_text = torch.cat(label_text, dim=1).to(args.device)
            panel = torch.cat((label_text, show(x, y), show(x_t, y)), dim=2)

            self.viewer.render(panel)

            self.total += target.size(0)
            self.correct += predicted.eq(target.cpu()).sum().item()
            running_loss = self.ll / self.batch_step
            accuracy = 100.0 * self.correct / self.total

            self.batch.set_description(f'Epoch: {epoch} {args.optim_class} LR: {get_lr(optim)} '
                                       f'{self.type} Loss: {running_loss:.12f} '
                                       f'Accuracy {accuracy:.4f}% {self.correct}/{self.total} of {self.len_dataset}')

            max_entropy_P = it.entropy(torch.ones_like(P) / P.numel())
            max_entropy_y = it.entropy(torch.ones(y.size(1)) / y.size(1))
            wandb_log = {
                f'{self.type}_loss': loss.item(),
                f'{self.type}_accuracy': accuracy,
                f'{self.type}_entropy_P_(max: {max_entropy_P})': it.entropy(P).item(),
                f'{self.type}_entropy_y_(max: {max_entropy_y})': torch.mean(it.entropy(F.softmax(y), dim=1)).item(),
                f'{self.type}_entropy_yt (max: {max_entropy_y})': torch.mean(it.entropy(F.softmax(y_t), dim=1)).item()
            }
            if self.batches == self.batch_step:
                wandb_log[f'{self.type}_final_results_panel'] = wandb.Image(panel)
            wandb.log(wandb_log)
            global_step += 1
            return accuracy, self.guesser, panel

    def log_epoch(confusion, best_precision, test_accuracy, train_accuracy):
        precis, ave_precis = precision(confusion)
        print('')
        print(f'{Fore.CYAN}RESULTS FOR EPOCH {Fore.LIGHTYELLOW_EX}{epoch}{Style.RESET_ALL}')
        for i, cls in enumerate(datapack.class_list):
            print(f'{Fore.LIGHTMAGENTA_EX}{cls} : {precis[i].item()}{Style.RESET_ALL}')
        best_precision = ave_precis if ave_precis > best_precision else best_precision
        print(f'{Fore.GREEN}ave precision : {ave_precis} best: {best_precision} test accuracy {test_accuracy} '
              f'train accuracy {train_accuracy}{Style.RESET_ALL}')
        return ave_precis, best_precision

    def nop(args, x, target):
        return x.to(args.device), target.to(args.device)

    def flatten(args, x, target):
        return x.flatten(start_dim=1).to(args.device), target.to(args.device)

    """ reproducibility """
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    """ variables """
    run_dir = f'data/models/classifiers/{args.dataset_name}/{args.model_name}/run_{args.run_id}'
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0
    ave_precision = 0.0
    best_precision = 0.0
    train_accuracy = 0.0
    test_accuracy = 0.0

    """ data """
    datapack = package.datasets[args.dataset_name]
    trainset, testset = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.dataroot)
    train = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)
    test = DataLoader(testset, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)
    # augment = flatten if args.model_type == 'fc' else nop

    augment = TpsAndRotateSecond(args.data_aug_tps_cntl_pts, args.data_aug_tps_variance, args.data_aug_max_rotate, padding_mode='border')

    """ model """
    encoder, meta = mnn.make_layers(args.model_encoder, args.model_type, LayerMetaData(datapack.shape))
    classifier = models.classifier.Classifier(encoder, meta, num_classes=datapack.num_classes).to(args.device)
    print(classifier)

    """ optimizer """
    optim, scheduler = config.get_optim(args, classifier.parameters())

    """ apex mixed precision """
    opt_level = 'O1'
    classifier, optim = amp.initialize(classifier, optim, opt_level=opt_level)

    if args.load is not None:
        checkpoint = torch.load(args.load)
        classifier.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
        # classifier.load_state_dict(torch.load(args.load))

    """ loss function """

    def IID_loss(x_out, x_tf_out, lamb=1.0):
        eps = torch.finfo(x_out.dtype).eps

        # has had softmax applied
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                                 k)  # but should be same, symmetric

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        p_i_j[(p_i_j < eps).data] = eps
        p_j[(p_j < eps).data] = eps
        p_i[(p_i < eps).data] = eps

        loss = - p_i_j * (torch.log(p_i_j)
                          - lamb * torch.log(p_j)
                          - lamb * torch.log(p_i))

        loss = loss.sum()

        loss_no_lamb = - p_i_j * (torch.log(p_i_j)
                                  - torch.log(p_j)
                                  - torch.log(p_i))

        loss_no_lamb = loss_no_lamb.sum()

        return loss, loss_no_lamb

    def compute_joint(x_out, x_tf_out):
        # produces variable that requires grad (since args require grad)

        bn, k = x_out.size()
        assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def mi_loss(x, x_t):
        """make a joint distribution from the batch """
        x = F.softmax(x, dim=1)
        x_t = F.softmax(x_t, dim=1)
        P = torch.matmul(x.T, x_t) / x.size(0)

        """symmetrical"""
        P = (P + P.T) / 2.0

        """return negative of mutual information as loss"""
        return - mi(P), P

    criterion = mi_loss

    def to_device(data, device):
        return tuple([x.to(device) for x in data])

    panel = None

    """ training/test loop """
    for i, epoch in enumerate(range(args.epochs)):

        batch = Batch('train', train, trainset)
        for data in batch:
            x, target = to_device(data, device=args.device)
            x, x_t, loss_mask = augment(x)

            # viewer.render(x_t)

            optim.zero_grad()
            y = classifier(x)
            y_t = classifier(x_t)
            loss, P = criterion(y, y_t)
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optim.step()

            train_accuracy, guesser, panel = batch.log_step()

            if i % args.checkpoint_freq == 0:
                checkpoint = {
                    'model': classifier.state_dict(),
                    'optimizer': optim.state_dict(),
                    'amp': amp.state_dict()
                }
                torch.save(checkpoint, run_dir + '/checkpoint_amp.pt')
                # torch.save(classifier.state_dict(), run_dir + '/checkpoint')
        with torch.no_grad():
            batch = Batch('test', test, testset)
            for data in batch:
                x, target = to_device(data, device=args.device)
                x, x_t, loss_mask = augment(x)

                y = classifier(x)
                y_t = classifier(x_t)
                loss, P = criterion(y, y_t)

                test_accuracy, guesser, panel = batch.log_step()

        ave_precision, best_precision = log_epoch(batch.confusion, best_precision, test_accuracy, train_accuracy)
        scheduler.step()

        if ave_precision >= best_precision:
            wandb.run.summary['best_precision'] = ave_precision
            # torch.save(classifier.state_dict(), run_dir + '/best')
            best = {
                'model': classifier.state_dict(),
                'optimizer': optim.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(best, run_dir + '/best_amp.pt')

    return ave_precision, best_precision, train_accuracy, test_accuracy


if __name__ == '__main__':
    """ configuration """
    args = config.config()
    pygame.init()
    wandb.init(project='iic', name=args.name)
    wandb.config.update(args)
    torch.cuda.set_device(args.device)
    main(args)
