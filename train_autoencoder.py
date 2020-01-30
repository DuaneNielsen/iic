import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from colorama import Fore, Style
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import statistics as stats
from models import mnn, autoencoder
from utils.viewer import UniImageViewer, make_grid
import datasets.package as package
from config import config

scale = 4
view_in = UniImageViewer('in', screen_resolution=(128 * 2 * scale, 128 * scale))
view_z = UniImageViewer('z', screen_resolution=(128//2 * 5 * scale, 128//2 * 4 * scale))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log(phase):
    writer.add_scalar(f'{phase}_loss', loss.item(), global_step)

    if args.display is not None and i % args.display == 0:
        recon = torch.cat((x[0], x_[0]), dim=2)
        writer.add_image(f'{phase}_recon', recon, global_step)

        if args.display:
            view_in.render(recon)

        if args.model_type != 'fc':
            latent = make_grid(z[0].unsqueeze(1), 4, 4)
            writer.add_image(f'{phase}_latent', latent.squeeze(0), global_step)
            if args.display:
                view_z.render(latent)


if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)

    """ variables """
    best_loss = 100.0
    run_dir = f'data/models/{args.dataset_name}/{args.model_name}/run_{args.run_id}'
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0

    """ data """
    datapack = package.datasets[args.dataset_name]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.dataroot)
    train_l = DataLoader(train, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder = mnn.make_layers(args.model_encoder, type=args.model_type)
    decoder = mnn.make_layers(args.model_decoder, type=args.model_type)

    if args.model_type == 'conv':
        auto_encoder = autoencoder.AutoEncoder(encoder, decoder, init_weights=args.load is None).to(args.device)
    elif args.model_type == 'fc':
        auto_encoder = autoencoder.LinearAutoEncoder(encoder, decoder, init_weights=args.load is None).to(args.device)
    else:
        raise Exception('model type string invalid')

    if args.load is not None:
        auto_encoder.load_state_dict(torch.load(args.load))

    """ optimizer """
    if args.optimizer == 'Adam':
        optim = Adam(auto_encoder.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optim = SGD(auto_encoder.parameters(), lr=args.lr)

    if args.scheduler is not None:
        scheduler = ReduceLROnPlateau(optim, mode='min')

    """ apex mixed precision """
    # if args.device != 'cpu':
    #     model, optimizer = amp.initialize(auto_encoder, optim, opt_level=args.opt_level)

    """ loss function """
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):

        """ training """
        batch = tqdm(train_l, total=len(train) // args.batchsize)
        for i, (x, _) in enumerate(batch):
            x = x.to(args.device)

            optim.zero_grad()
            z, x_ = auto_encoder(x)
            loss = criterion(x_, x)
            if not args.demo:
                loss.backward()
                optim.step()

            batch.set_description(f'Epoch: {epoch} {args.optimizer} LR: {get_lr(optim)} Train Loss: {loss.item()}')

            log('train')

            if i % args.checkpoint_freq == 0 and args.demo == 0:
                torch.save(auto_encoder.state_dict(), run_dir + '/checkpoint')

            global_step += 1

        """ test  """
        with torch.no_grad():
            ll = []
            batch = tqdm(test_l, total=len(test) // args.batchsize)
            for i, (x, _) in enumerate(batch):
                x = x.to(args.device)

                z, x_ = auto_encoder(x)
                loss = criterion(x_, x)

                batch.set_description(f'Epoch: {epoch} Test Loss: {loss.item()}')
                ll.append(loss.item())
                log('test')

                global_step += 1

        """ check improvement """
        ave_loss = stats.mean(ll)
        if args.scheduler is not None:
            scheduler.step(ave_loss)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        """ save if model improved """
        if ave_loss <= best_loss and not args.demo:
            torch.save(auto_encoder.state_dict(), run_dir + '/best')


