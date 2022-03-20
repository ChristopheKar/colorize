import os
import argparse

import torch
from torch.utils.data import DataLoader

from utils import LabImages
from utils import save_image
from models import ECCV16Model, SIGGRAPH17Model


def colorize(
    image_in,
    output,
    model='siggraph17',
    weights=None,
    caffe=False,
    recursive=True):

    target_size = (256, 256)

    # Load selected model and checkpoint
    if (model == 'eccv16'):
        # Only ECCV16 model has Caffe version
        if (caffe):
            target_size = (224, 224)

        model = ECCV16Model(checkpoint=weights, caffe=caffe)

    elif (model == 'siggraph17'):
        model = SIGGRAPH17Model(checkpoint=weights)

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
                output_path = get_output_path(img_path, output)
                save_image(image, output_path)

    print(f'\nSaved images in `{os.path.dirname(output_path)}/`.')


def get_output_path(img_path, output_path):

    img_name = img_path.split('/')[-1]
    img_ext = img_name.rsplit('.', 1)[-1]
    if (output_path is not None):
        filepath = os.path.join(output_path, img_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        filepath = img_path.replace(f'.{img_ext}', f'_colorized.{img_ext}')

    return filepath


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
        '-w', '--weights',
        type=str, required=False,
        default=True,
    	help=(
            'Path to model weights, if not specified, '
            'model uses default weights from model zoo.')
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
        model=args.model,
        weights=args.weights,
        caffe=args.caffe,
        recursive=args.recursive)
