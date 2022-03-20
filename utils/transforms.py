import numpy as np
import cv2

import torch
import torch.nn.functional as F


class Compose(object):

    """Composes several transforms together.
    Taken from torchvision.transforms.Compose and customized to allow
    for a flexible number of inputs and outputs.

    Parameters
    ----------
    transforms: list of objects
        List of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, *args):
        for transform in self.transforms:
            if (isinstance(args, list) or (isinstance(args, tuple))):
                args = transform(*args)
            else:
                args = transform(args)
        return args


    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ReadImage:

    """Read image using OpenCV as RGB."""

    def __init__(self, read_mode=1):
        self.read_mode = read_mode


    def __call__(self, image_path):

        # Read image with OpenCV
        image = cv2.imread(image_path, self.read_mode)
        # Convert from grayscale to RGB if image is 2D
        if (image.ndim == 2):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Convert from BGR to RGB if image is 3D
        elif (image.ndim == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                'Image must have 2 or 3 dimensions, got {image.ndim}.')

        image = (image/255.).astype('float32')

        return image


class PadResize:

    """Resize image keeping aspect ratio, and pad to fill target size.

    Parameters
    ----------
    target_size: tuple of int, optional, default=(256, 256)
        Target size for output image, as (width, height).
    pad_val: None or number or tuple of number,
                optional, default=255
        Value to use when padding, if None, no padding is applied.
    keep_ratio: bool, optional, default=True
        If True, keeps aspect ration when resizing. If True and `pad_val`
        is set to None, the output size might be different from `target_size`.
    """

    def __init__(
        self, target_size=(256, 256), pad_val=255, keep_ratio=True):

        self.target_w = target_size[0]
        self.target_h = target_size[1]
        self.target_size = (self.target_w, self.target_h)
        self.pad_value = pad_val
        self.keep_ratio = keep_ratio


    def __call__(self, image):

        if (self.keep_ratio):
            # Compute minimum and maximum dimensions
            img_size = (image.shape[1], image.shape[0])
            max_dim_idx = np.argmax(img_size)
            min_dim_idx = int(not max_dim_idx)
            max_dim = img_size[max_dim_idx]
            min_dim = img_size[min_dim_idx]
            # Compute resize ratio
            ratio = self.target_size[max_dim_idx]/max_dim
            # Compute new target size
            # target size for max dimension stays the same
            # target size for min dimension is changed to min dimension * ratio
            target_size = list(self.target_size)
            target_size[min_dim_idx] = int(min_dim*ratio)
        else:
            target_size = self.target_size

        # Resize image
        output_image = cv2.resize(
            image, target_size, interpolation=cv2.INTER_AREA)

        if (self.pad_value is not None):
            # Compute pad sizes
            max_dim = self.target_size[max_dim_idx]
            img_size = (output_image.shape[1], output_image.shape[0])
            pad_left, pad_top = [(max_dim - dim)//2 for dim in img_size]
            pad_right = max_dim - (img_size[0] + pad_left)
            pad_bottom = max_dim - (img_size[1] + pad_top)

            # Pad image
            output_image = cv2.copyMakeBorder(
                output_image,
                pad_top, pad_bottom,
                pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=self.pad_value)

        return output_image


class ResizeToMatch:

    """Resize image tensor to match another's spatial dimensions.

    Parameters
    ----------
    mode: str, optional, default='bilinear'
        Interpolation mode.

    """

    def __init__(self, mode='bilinear'):
        self.mode = mode


    def __call__(self, x, y):
        """Interpolate tensor `y` to match spatial dimensions of tensor `x`.

        Parameters
        ----------
        x: torch.Tensor
            Target tensor to match.
        y: torch.Tensor
            Tensor to interpolate to match `x`.

        Returns
        """
        target_size = x[0].shape[-2:]
        curr_size = y[1].shape[-2:]
        if (np.any(target_size != curr_size)):
            y = F.interpolate(
                y.expand((1, -1, -1, -1)),
                size=target_size, mode=self.mode).squeeze()

        return (x, y)


class TransformColorspace:

    """Perform a colorspace transformation using OpenCV.

    Parameters
    ----------
    _from: str
        Input colorspace.
    _to: str
        Target colorspace.
    keep_channels: None or int or slice
        If None, returns entire image, otherwise returns
        only returns the image with the selected channel(s).

    """

    def __init__(self, _from, _to, keep_channels=None):

        self.allowed_transformations = [
            'COLOR_RGB2LAB',
            'COLOR_LAB2RGB',
            'COLOR_BGR2LAB',
            'COLOR_LAB2BGR',
            'COLOR_RGB2BGR',
            'COLOR_BGR2RGB',
            'COLOR_GRAY2BGR',
            'COLOR_GRAY2RGB',
            'COLOR_BGR2GRAY',
            'COLOR_RGB2GRAY',
            'COLOR_GRAY2LAB',
            'COLOR_LAB2GRAY'
        ]
        attr = f'COLOR_{_from.upper()}2{_to.upper()}'
        if (attr in self.allowed_transformations):
            self.transform = getattr(cv2, attr)
        else:
            raise ValueError(f'{attr} is not an allowed transformation.')

        self.channels = keep_channels


    def __call__(self, image):
        if (self.channels is not None):
            return cv2.cvtColor(image, self.transform)[:, :, self.channels]
        else:
            return cv2.cvtColor(image, self.transform)


class ConcatenateChannels:

    """Concatenate two images by channel.

    Parameters
    ----------
    dim: int, optional, default=0
        Channel dimension used for concatenation.
    """

    def __init__(self, dim=0):
        self.dim = dim


    def __call__(self, x, y):
        """Concatenate tensors `x` and `y` on `dim`."""
        return torch.cat((x, y), dim=self.dim)


class ToTensor:

    """Convert image from `numpy.ndarray` to `torch.Tensor`."""

    def __call__(self, image):

        tensor = torch.Tensor(image)
        if (tensor.ndim == 2):
            tensor = tensor.expand((1, -1, -1))
        elif (tensor.ndim == 3):
            tensor = tensor.transpose(2, 0).transpose(1, 2)
        else:
            raise ValueError(f'Unknown shape {tensor.ndim}.')

        return tensor


class ToNumpy:

    """Convert image from `torch.Tensor` to `numpy.ndarray`."""

    def __call__(self, image):
        return image.data.squeeze().cpu().numpy().transpose((1, 2, 0))


class ToImage:

    """Convert image from float representation to uint8."""

    def __call__(self, image):
        if ('float' in image.dtype.name):
            return (np.clip(image, 0, 1)*255).astype('uint8')
        else:
            return image
