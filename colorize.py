import os
import argparse

import torch
from torch.utils.data import DataLoader

from utils import LabImages
from utils import save_image
from models import ECCV16Model, SIGGRAPH17Model


def colorize(image_in, output, model='siggraph17', caffe=False, recursive=True):

    target_size = (256, 256)

    # Load selected model and checkpoint
    if (model == 'eccv16'):

        # Only ECCV16 model has Caffe version
        if (caffe):
            checkpoint = dict(
                prototxt='weights/caffe/colorization_deploy_v2.prototxt',
                points='weights/caffe/pts_in_hull.npy',
                model='weights/caffe/colorization_release_v2.caffemodel'
            )
            target_size = (224, 224)
        else:
            checkpoint = dict(model='weights/eccv16-9b330a0b.pth')

        model = ECCV16Model(pretrained=checkpoint, caffe=caffe)

    elif (model == 'siggraph17'):
        checkpoint = dict(model='weights/siggraph17-df00044c.pth')
        model = SIGGRAPH17Model(pretrained=checkpoint)

    # Load dataset
    dataset = LabImages(
        image_in, recursive=recursive, target_size=target_size, caffe=caffe)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, batch in enumerate(loader):
        print(f'Processing image {idx+1}/{len(loader)}', end='\r')

        with torch.no_grad():
            # Forward pass
            batch['pred_ab'] = model.forward(batch['resized_l'])
            # Post-process predicted images and save output
            batch = zip(
                batch['image_path'], batch['original_l'], batch['pred_ab'])
            for img_path, image_l, image_ab in batch:
                image = dataset.output_transform(image_l, image_ab)
                filepath = os.path.join(output, img_path.split('/')[-1])
                save_image(image, filepath)
    print()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str, required=True,
    	help=(
            'Path to input image, directory of images, '
            'or text file containing paths to images (jpg or png).')
    )
    parser.add_argument(
        '-o', '--output',
        type=str, required=False,
        default=None,
    	help='Path to output directory.')
    parser.add_argument(
        '-r', '--recursive',
        action='store_true', required=False,
    	help='Explore input directory recursively.')
    parser.add_argument(
        '-m', '--model',
        type=str, required=False,
        default='siggraph17',
    	help=(
            'Model to use for colorization. '
            'Available models are ECCV16 and SIGGRAPH17.')
    )
    parser.add_argument(
        '--caffe',
        action='store_true', required=False,
    	help='Use Caffe version for ECCV16 model.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    colorize(
        args.input, args.output,
        model=args.model, caffe=args.caffe,
        recursive=args.recursive)
