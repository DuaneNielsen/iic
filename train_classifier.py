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


def mi(P):
    eps = torch.finfo(P.dtype).eps
    P[P < eps] = eps
    m0 = torch.sum(P, dim=0, keepdim=True)
    m1 = torch.sum(P, dim=1, keepdim=True)
    return torch.sum((P.log2() - m0.log2() - m1.log2()) * P)


def main(args):
    def precision(confusion):
        correct = confusion * torch.eye(confusion.shape[0])
        incorrect = confusion - correct
        correct = correct.sum(0)
        incorrect = incorrect.sum(0)
        precision = correct / (correct + incorrect)
        total_correct = correct.sum().item()
        total_incorrect = incorrect.sum().item()
        percent_correct = total_correct / (total_correct + total_incorrect)
        return precision, percent_correct

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    class Batch:
        def __init__(self, type, loader, dataset):
            self.type = type
            self.loader = loader
            self.batch = tqdm(loader, total=len(dataset) // args.batchsize)
            self.ll = 0
            self.confusion = torch.zeros(datapack.num_classes, datapack.num_classes)
            self.total = 0
            self.correct = 0
            self.batch_step = 0

        def __iter__(self):
            return iter(self.batch)

        def log_step(self):
            self.batch_step += 1
            self.ll += loss.detach().item()

            _, predicted = y.detach().max(1)
            self.total += target.size(0)
            self.correct += predicted.eq(target).sum().item()
            running_loss = self.ll / self.batch_step
            accuracy = 100.0 * self.correct / self.total

            self.batch.set_description(f'Epoch: {epoch} {args.optim_class} LR: {get_lr(optim)} '
                                       f'{self.type} Loss: {running_loss:.4f} '
                                       f'Accuracy {accuracy:.4f}% {self.correct}/{self.total}')

            if self.type == 'test':
                for p, t in zip(predicted, target):
                    self.confusion[p, t] += 1

            writer.add_scalar(f'{id}_loss', loss.item(), global_step)
            writer.add_scalar(f'{id}_accuracy', accuracy, global_step)
            return accuracy

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
    augment = flatten if args.model_type == 'fc' else nop

    """ model """
    encoder, meta = mnn.make_layers(args.model_encoder, args.model_type, LayerMetaData(datapack.shape))
    classifier = models.classifier.Classifier(encoder, meta, num_classes=datapack.num_classes).to(args.device)
    print(classifier)

    if args.load is not None:
        classifier.load_state_dict(torch.load(args.load))

    """ optimizer """
    optim, scheduler = config.get_optim(args, classifier.parameters())

    """ loss function """
    criterion = nn.CrossEntropyLoss()

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

    def loss(x, x_t):

        x1 = F.softmax(x, dim=1)
        x2 = F.softmax(x1, dim=1)
        P = torch.matmul(x1.T, x2) / x1.size(0)
        P = (P + P.T) / 2.0
        return - mi(P)

    """ training/test loop """
    for i, epoch in enumerate(range(args.epochs)):

        batch = Batch('train', train, trainset)
        for x, target in batch:
            x, target = augment(args, x, target)

            optim.zero_grad()
            y = classifier(x)
            loss = criterion(y, target)
            loss.backward()
            optim.step()

            train_accuracy = batch.log_step()

            if i % args.checkpoint_freq == 0:
                torch.save(classifier.state_dict(), run_dir + '/checkpoint')

        batch = Batch('test', test, testset)
        for x, target in batch:
            x, target = augment(args, x, target)

            y = classifier(x)
            loss = criterion(y, target)

            test_accuracy = batch.log_step()

        ave_precision, best_precision = log_epoch(batch.confusion, best_precision, test_accuracy, train_accuracy)
        scheduler.step()

        if ave_precision >= best_precision:
            torch.save(classifier.state_dict(), run_dir + '/best')

    return ave_precision, best_precision, train_accuracy, test_accuracy


if __name__ == '__main__':
    """  configuration """
    args = config.config()
    main(args)
