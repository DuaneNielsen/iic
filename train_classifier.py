import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style
import torch.nn as nn
import models.classifier
from models import mnn
import config
from datasets import package
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import numpy as np


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
    encoder, output_shape = mnn.make_layers(args.model_encoder, args.model_type, input_shape=datapack.shape)
    classifier = models.classifier.Classifier(encoder, output_shape, num_classes=datapack.num_classes).to(args.device)
    print(classifier)

    if args.load is not None:
        classifier.load_state_dict(torch.load(args.load))

    """ optimizer """
    optim, scheduler = config.get_optim(args, classifier.parameters())

    """ loss function """
    criterion = nn.CrossEntropyLoss()

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
