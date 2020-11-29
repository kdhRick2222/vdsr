## loss function, 스케일 하나만!
import argparse
import os

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model import VDSR
from datasets import *
from utils import PSNR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = VDSR().to(device)

    criterion = nn.MSELoss()

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset,batch_size=1)

    for epoch in range(args.num_epochs):
        model.train()
        loss_sum = 0

        if epoch<20:
            learning_rate = args.lr
        elif epoch>=20 and epoch<40:
            learning_rate = 0.001
        elif epoch>=40 and epoch<60:
            learning_rate = 1e-2 * args.lr
        else:
            learning_rate = 1e-3 * args.lr

        print("lr = {}".format(learning_rate))

        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            
        print("TRAIN: EPOCH %04d / %04d | LOSS %.4f" %
              (epoch, args.num_epochs, loss_sum / 64))

        model.eval()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            psnr = PSNR(preds, labels)

            print("Eval_PSNR = {}".format(psnr))
            
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))









