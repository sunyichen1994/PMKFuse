import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader.msrs_data import MSRS_data
from models.common import gradient, clamp
from models.fusion_model import Global_Encoder, Local_Encoder, PMKFuse
from loss import Fusionloss, cc, loss_Qabf,LossIn,LossGrad




def init_seeds(seed=3407):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PMKFuse')
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/msrs_train1',  #msrs_train1
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='pretrained')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
    parser.add_argument('--loss_weight', default='[1,100,100]', type=str,
                        metavar='N', help='loss weight')
    parser.add_argument('--seed', default=3407, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()
    init_seeds(args.seed)

    train_dataset = MSRS_data(args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.arch == 'fusion_model':
        model = PMKFuse()
        model = model.cuda()

        if args.cuda and torch.cuda.device_count() > 1:
             model = nn.DataParallel(model)


        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.start_epoch, args.epochs):
            if epoch < args.epochs // 6:
                lr = args.lr
            else:
                lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)


            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            for vis_image, vis_y_image, _, _, inf_image, _ in train_tqdm:
                vis_y_image = vis_y_image.cuda()
                vis_image = vis_image.cuda()
                inf_image = inf_image.cuda()
                optimizer.zero_grad()
                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)



                loss_in_module = LossIn()
                loss_grad_module = LossGrad()

                global_encoder = Global_Encoder().to(device)
                local_encoder = Local_Encoder().to(device)


                feature_V_G, feature_I_G = global_encoder(vis_y_image, inf_image)
                feature_V_L, feature_I_L = local_encoder(vis_y_image, inf_image)

                cc_loss_G = cc(feature_V_G, feature_I_G)
                cc_loss_L = cc(feature_V_L, feature_I_L)

                loss_decomp = (cc_loss_G) ** 2 / (1.001 + cc_loss_L)

                lossin = loss_in_module(vis_y_image, inf_image, fused_image)
                lossgrad = loss_grad_module(vis_y_image, inf_image, fused_image)

                t1, t2, t3 = eval(args.loss_weight)

                loss = t1 * lossin + t2 * lossgrad + t3 * loss_decomp

                train_tqdm.set_postfix(epoch=epoch,
                                       loss_in=t1 * lossin.item(),
                                       loss_grad=t2 * lossgrad.item(),
                                       loss_decomp=t3 * loss_decomp.item(),
                                       loss_total=loss.item())
                loss.backward()
                optimizer.step()

            torch.save(model.state_dict(), f'{args.save_path}/fusion_model_epoch_{epoch}.pth')




