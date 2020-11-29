import argparse
import os

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from model import VDSR
from datasets import *
from utils import PSNR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file2', type=str, required=True)
    parser.add_argument('--train_file3', type=str, required=True)
    parser.add_argument('--train_file4', type=str, required=True)
    parser.add_argument('--eval_file2', type=str, required=True)
    parser.add_argument('--eval_file3', type=str, required=True)
    parser.add_argument('--eval_file4', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=str, default="multi")
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

    train_dataset2 = TrainDataset(args.train_file2)
    train_dataset3 = TrainDataset(args.train_file3)
    train_dataset4 = TrainDataset(args.train_file4)
    train_dataset = train_dataset2 + train_dataset3 + train_dataset4
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset2 = EvalDataset(args.eval_file2)
    eval_dataset3 = EvalDataset(args.eval_file3)
    eval_dataset4 = EvalDataset(args.eval_file4)
    eval_dataset = eval_dataset2 + eval_dataset3 + eval_dataset4
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=1)

    for epoch in range(args.num_epochs):
        model.train()
        loss_arr = []

        if epoch < 20:
            learning_rate = args.lr
        elif epoch >= 20 and epoch < 40:
            learning_rate = 0.1 * args.lr
        elif epoch >= 40 and epoch < 60:
            learning_rate = 0.01 * args.lr
        else:
            learning_rate = 1e-3 * args.lr

        print("lr = {}".format(learning_rate))

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())

            loss_arr += [loss.item()]

        print("TRAIN: EPOCH %04d / %04d | LOSS %.4f" %
              (epoch, args.num_epochs, np.mean(loss_arr)))

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

            