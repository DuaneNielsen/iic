import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style
import torch.nn as nn
import statistics as stats
import models.classifier
from models import vgg, mnn
import config
from datasets import package
from tensorboardX import SummaryWriter


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
        self.ll = []
        self.confusion = torch.zeros(num_classes, num_classes)
        self.total = 0
        self.correct = 0

    def __iter__(self):
        return iter(self.batch)

    def log_step(self):
        global optim
        self.ll.append(loss.detach().item())

        _, predicted = y.detach().max(1)
        self.total += target.size(0)
        self.correct += predicted.eq(target).sum().item()

        if self.type == 'train':
            self.batch.set_description(f'Epoch: {epoch} {args.optim_class} LR: {get_lr(optim)} '
                                       f'Train Loss: {loss.item()} '
                                       f'Accuracy {100.0 * self.correct/self.total}% {self.correct}/{self.total}')

        if self.type == 'test':
            self.batch.set_description(f'Epoch: {epoch} Test Loss: {stats.mean(self.ll)} '
                                       f'Train Loss: {loss.item()} '
                                       f'Accuracy {100.0 * self.correct/self.total}% {self.correct}/{self.total}')

            for p, t in zip(predicted, target):
                self.confusion[p, t] += 1

        writer.add_scalar(f'{id}_loss', loss.item(), global_step)


def log_epoch(confusion):
    global best_precision
    precis, ave_precis = precision(confusion)
    print('')
    print(f'{Fore.CYAN}RESULTS FOR EPOCH {Fore.LIGHTYELLOW_EX}{epoch}{Style.RESET_ALL}')
    for i, cls in enumerate(datapack.class_list):
        print(f'{Fore.LIGHTMAGENTA_EX}{cls} : {precis[i].item()}{Style.RESET_ALL}')
    best_precision = ave_precis if ave_precis > best_precision else best_precision
    print(f'{Fore.GREEN}ave precision : {ave_precis} best: {best_precision} {Style.RESET_ALL}')
    return ave_precis


if __name__ == '__main__':

    """  configuration """
    args = config.config()

    """ variables """
    run_dir = f'data/models/classifiers/{args.dataset_name}/{args.model_name}/run_{args.run_id}'
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0
    best_precision = 0.0

    """ data """
    datapack = package.datasets[args.dataset_name]
    num_classes = len(datapack.class_list)
    trainset, testset = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.dataroot)
    train = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)
    test = DataLoader(testset, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder, output_shape = mnn.make_layers(args.model_encoder, input_shape=datapack.hw)
    classifier = models.classifier.Classifier(encoder, output_shape, num_classes=num_classes, init_weights=True).to(args.device)
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
            x, target = x.to(args.device), target.to(args.device)

            optim.zero_grad()
            y = classifier(x)
            loss = criterion(y, target)
            loss.backward()
            optim.step()

            batch.log_step()

            if i % args.checkpoint_freq == 0:
                torch.save(classifier.state_dict(), run_dir + '/checkpoint')

        batch = Batch('test', test, testset)
        for x, target in batch:
            x, target = x.to(args.device), target.to(args.device)

            y = classifier(x)
            loss = criterion(y, target)

            batch.log_step()

        ave_precis = log_epoch(batch.confusion)
        scheduler.step()

        if ave_precis >= best_precision:
            torch.save(classifier.state_dict(), run_dir + '/best')
