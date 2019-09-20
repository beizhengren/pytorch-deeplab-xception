# 
# demo.py 
# 
import argparse
import os
import numpy as np
import logging
import sys
import os.path

from PIL import Image
from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image


def blend_two_images(img1_path, img2_path, output_path):
    img1 = Image.open( img1_path)
    #    if not img1.exists()
    img1 = img1.convert('RGBA')

    img2 = Image.open( img2_path)
    img2 = img2.convert('RGBA')

    r, g, b, alpha = img2.split()
    alpha = alpha.point(lambda i: i>0 and 204)

    img = Image.composite(img2, img1, alpha)
    if output_path == "result":
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    # img.show()
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, required=True, help='directory of images to test')
    parser.add_argument('--out-path', type=str, required=True, help='directory of mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=True, 
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    model = DeepLab(num_classes=3,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    print(f"The args.ckpt is : {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    
    for img_path in os.listdir(args.in_path):
        if os.path.splitext(img_path)[-1] not in ['.jpg']:
            print('skip {}'.format(img_path))
            continue
        img_path = os.path.join(args.in_path, img_path)
        output_path = os.path.join(args.out_path, os.path.splitext(os.path.split(img_path)[-1])[-2] + "-seg" + ".jpg")
        # print("output path is {}".format(output_path))
        combine_path =  os.path.join(args.out_path, os.path.splitext(os.path.split(img_path)[-1])[-2] + "-blend" + ".png")
        # print("blend path is {}".format(combine_path))  
        image = Image.open(img_path).convert('RGB')
        target = Image.open(img_path).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            image = image.cuda()
        with torch.no_grad():
            output = model(tensor_in)
        
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                            3, normalize=False, range=(0, 255))
        print("type(grid) is:{}".format( type(grid_image)))
        print("grid_image.shape is:{}".format( grid_image.shape))
        save_image(grid_image, output_path)
        print("saved {}".format(output_path))
        blend_two_images(img_path, output_path, combine_path)
        print("blended {}\n".format(combine_path))

if __name__ == "__main__":
   main()
