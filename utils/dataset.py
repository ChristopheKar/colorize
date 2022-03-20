import os
from torch.utils.data import Dataset

from . import transforms


class LabImages(Dataset):

    def __init__(
        self, image_in,
        recursive=True,
        target_size=(256, 256),
        caffe=False):

        self.caffe = caffe
        self.allowed_exts = ('jpeg', 'jpg', 'png')
        self.image_in = image_in
        self.image_paths = []

        raise_error = False

        # Parse image input

        # Input is a directory
        if (os.path.isdir(image_in)):
            # Get image paths from directory
            if (recursive):
                for root, dirs, files in os.walk(image_in):
                    for name in files:
                        if (name.lower().endswith(self.allowed_exts)):
                            img_path = os.path.join(root, name)
                            self.image_paths.append(img_path)
            else:
                for name in os.listdir(image_in):
                    if (name.lower().endswith(self.allowed_exts)):
                        img_path = os.path.join(image_in, name)
                        self.image_paths.append(img_path)

        # Input is a file
        elif (os.path.isfile(image_in)):

            # Read image paths from text file
            if (image_in.lower().endswith('.txt')):
                with open(image_in, 'r') as fobj:
                    self.image_paths = fobj.read().strip().splitlines()

            # Input is an image file
            elif (image_in.lower().endswith(self.allowed_exts)):
                self.image_paths = [image_in]

            else:
                raise_error = True

        else:
            raise_error = True

        if (raise_error):
            err = (
                'Parameter `image_in` must be a directory containing '
                'images, or a text file containing paths to images, '
                'or a path to an image file.\n'
                f'Supported image extensions: {" ".join(self.allowed_exts)}'
            )
            raise OSError(err)

        # Set target size
        self.target_size = target_size

        # Define transforms

        # Transform for resized `L` channel
        self.resize_l = transforms.Compose([
            transforms.PadResize(
                self.target_size, keep_ratio=False, pad_val=None),
            transforms.TransformColorspace(
                _from='RGB', _to='LAB', keep_channels=0),
            transforms.ToTensor()
        ])
        # Transform for original-size `L` channel
        self.transform_l = transforms.Compose([
            transforms.TransformColorspace(
                _from='RGB', _to='LAB', keep_channels=0),
            transforms.ToTensor()
        ])
        # Transform for output
        self.output_transform = transforms.Compose([
            transforms.ResizeToMatch(mode='bilinear'),
            transforms.ConcatenateChannels(),
            transforms.ToNumpy(),
            transforms.TransformColorspace(
                _from='LAB', _to='RGB'),
            transforms.ToImage()
        ])


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image = transforms.ReadImage()(self.image_paths[idx])
        return {
            'image_path': self.image_paths[idx],
            'original_l': self.transform_l(image),
            'resized_l': self.resize_l(image)
        }
