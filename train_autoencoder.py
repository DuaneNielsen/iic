import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from colorama import Fore, Style
import torch.nn as nn
from iic.models import autoencoder, mnn
from iic.models.layerbuilder import LayerMetaData
from iic.utils.viewer import UniImageViewer, make_grid
import datasets.package as package
from iic import config
import torch.backends.cudnn
import numpy as np

scale = 4
view_in = UniImageViewer('in', screen_resolution=(128 * 2 * scale, 128 * scale))
view_z = UniImageViewer('z', screen_resolution=(128 // 2 * 5 * scale, 128 // 2 * 4 * scale))


def main(args):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def log(phase):
        writer.add_scalar(f'{phase}_loss', loss.item(), global_step)

        if args.display is not None and i % args.display == 0:
            recon = torch.cat((reverse_augment(x[0]), reverse_augment(x_[0])), dim=2)
            writer.add_image(f'{phase}_recon', recon, global_step)

            if args.display:
                view_in.render(recon)

            if args.model_type != 'fc':
                latent = make_grid(z[0].unsqueeze(1), 4, 4)
                writer.add_image(f'{phase}_latent', latent.squeeze(0), global_step)
                if args.display:
                    view_z.render(latent)

    def nop(x):
        return x

    def flatten(x):
        return x.flatten(start_dim=1)

    def reverse_flatten(x):
        return x.reshape(1, 28, 28)

    torch.cuda.set_device(args.device)

    """ reproducibility """
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    """ variables """
    best_loss = 100.0
    run_dir = f'data/models/autoencoders/{args.dataset_name}/{args.model_name}/run_{args.run_id}'
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0

    """ data """
    datapack = package.datasets[args.dataset_name]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.dataroot)
    train_l = DataLoader(train, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder, meta = mnn.make_layers(args.model_encoder, type=args.model_type, meta=LayerMetaData(datapack.shape))
    decoder, meta = mnn.make_layers(args.model_decoder, type=args.model_type, meta=meta)
    auto_encoder = autoencoder.AutoEncoder(encoder, decoder).to(args.device)
    print(auto_encoder)
    augment = flatten if args.model_type == 'fc' else nop
    reverse_augment = reverse_flatten if args.model_type == 'fc' else nop

    if args.load is not None:
        auto_encoder.load_state_dict(torch.load(args.load))

    """ optimizer """
    optim, scheduler = config.get_optim(args, auto_encoder.parameters())

    """ apex mixed precision """
    # if args.device != 'cpu':
    #     model, optimizer = amp.initialize(auto_encoder, optim, opt_level=args.opt_level)

    """ loss function """
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):

        """ training """
        batch = tqdm(train_l, total=len(train) // args.batchsize)
        for i, (x, _) in enumerate(batch):
            x = augment(x).to(args.device)

            optim.zero_grad()
            z, x_ = auto_encoder(x)
            loss = criterion(x_, x)
            if not args.demo:
                loss.backward()
                optim.step()

            batch.set_description(f'Epoch: {epoch} {args.optim_class} LR: {get_lr(optim)} Train Loss: {loss.item()}')

            log('train')

            if i % args.checkpoint_freq == 0 and args.demo == 0:
                torch.save(auto_encoder.state_dict(), run_dir + '/checkpoint')

            global_step += 1

        """ test  """
        with torch.no_grad():
            ll = 0.0
            batch = tqdm(test_l, total=len(test) // args.batchsize)
            for i, (images, _) in enumerate(batch):
                x = augment(images).to(args.device)

                z, x_ = auto_encoder(x)
                loss = criterion(x_, x)

                ll += loss.item()
                ave_loss = ll / (i + 1)
                batch.set_description(f'Epoch: {epoch} Test Loss: {ave_loss}')

                log('test')

                global_step += 1

        """ check improvement """
        scheduler.step(ave_loss)

        best_loss = ave_loss if ave_loss <= best_loss else best_loss
        print(f'{Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {best_loss} {Style.RESET_ALL}')

        """ save if model improved """
        if ave_loss <= best_loss and not args.demo:
            torch.save(auto_encoder.state_dict(), run_dir + '/best')

    return best_loss


if __name__ == '__main__':
    args = config.config()
    main(args)
