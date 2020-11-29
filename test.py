import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image

from model import VDSR
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--image_file', type=str, required=True)
    parser.add_argument('--realimage_file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=1)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = VDSR().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    # model, optim = torch.load(model.state_dict(), os.path.join(args.weights_file, 'epoch_150.pth'))

    model.eval()

    image = Image.open(args.realimage_file).convert('RGB')
    resample = Image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height))
    image = image.resize((image.width // args.scale, image.height // args.scale))
    image = image.resize((image.width * args.scale, image.height * args.scale))
    ## image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    resample = resample.resize((image_width, image_height))
    resample = np.array(resample).astype(np.float32)
    resample_1 = convert_rgb_to_ycbcr(resample)
    resample_2 = resample_1[..., 0]
    resample_2 /= 255.
    resample_2 = torch.from_numpy(resample_2).to(device)
    resample_2 = resample_2.unsqueeze(0).unsqueeze(0)

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(resample_2).clamp(0.0, 1.0)

   # psnr1 = PSNR(y, preds)
    print(y.shape, '|', preds.shape, '|',resample_2.shape)
    psnr1 = PSNR(preds, y)
    print('PSNR: {:.2f}'.format(psnr1))
    

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = Image.fromarray(output)
    output.save(args.image_file.replace('.', '_vdsr_x{}.'.format(args.scale)))