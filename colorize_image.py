import os
import argparse
import numpy as np
import cv2


def load_caffe_model(prototxt_path, model_path, points_path):
    """Load serialized Caffe colorization model"""
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(points_path)
    # Add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net


def preprocess_image(image, target_size=(224, 224)):
    """Rescale and resize image."""
    # Rescale pixels intensities in range [0, 1]
    scaled = image.astype("float32") / 255.0
    # Convert to Lab colorspace
    image_lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    # Resize image to target size
    resized_lab = cv2.resize(image_lab, target_size)

    return image_lab, resized_lab


def colorize_image(net, image_lab):
    """Colorize image."""
    # Extract 'L' channel
    image_L =  cv2.split(image_lab)[0]
    # Center channel (subtract mean)
    image_L = image_L - np.mean(image_L)
    # Model takes L channel as input
    net.setInput(cv2.dnn.blobFromImage(image_L))
    # Model predicts a and b channels
    image_ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    return image_ab


def postprocess_image(original_lab, image_ab):
    """Construct colorized image from predicted channels."""
    # resize the predicted channels to original size
    ab = cv2.resize(image_ab, (original_lab.shape[1], original_lab.shape[0]))
    # Colorized image is original L channels
    # and predicted resized ab channels
    image_L = cv2.split(original_lab)[0]
    colorized = np.concatenate((image_L[:, :, np.newaxis], ab), axis=2)
    # Convert to BGR colorspace
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    # Clip any values outside [0, 1] range
    colorized = np.clip(colorized, 0, 1)
    # Convert image from [0, 1] float to [0, 255] uint8
    colorized = (255 * colorized).astype("uint8")

    return colorized


def find_images(in_path, img_exts=('.png', '.jpg', '.jpeg'), verbose=1):
    """Return list of image paths parsed from input file or dir. path."""
    imgpaths = []
    # Input path is image path
    if (in_path.endswith(img_exts)):
        if (os.path.isfile(in_path)):
            imgpaths.append(in_path)
        else:
            raise OSError('file  at `{}` not found'.format(in_path))
    # Input path is directory
    elif (os.path.isdir(in_path)):
        # Find valid images in directory
        for file in os.listdir(in_path):
            if (file.endswith(img_exts)):
                imgpaths.append(os.path.join(in_path, file))
        if (verbose):
            print('Found {:d} images in directory.'.format(len(imgpaths)))
    else:
        msg = 'input path must be path to image or directory'
        raise ValueError(msg)

    return imgpaths


def read_image(filepath):
    """Read image from disk using OpenCV."""
    return cv2.imread(filepath)


def save_image(image, in_path, out_path):
    """
    Write image to disk using OpenCV.
    Image gets saved in output directory if provided
    or in input directory with original image name and
    `_colorized` appended at the end.
    """
    if (out_path is None):
        # Output path is same as input image path
        # but with `_colorized` appended at the end
        ext = '.' + in_path.split('.')[-1]
        filepath = in_path.replace(ext, '_colorized' + ext)
    elif (isinstance(out_path, str)):
        # Create output directory if it does not exist
        os.makedirs(out_path, exist_ok=True)
        filepath = os.path.join(out_path, in_path.split('/')[-1])
    else:
        raise TypeError('output path must be a string')

    # Save image
    print(filepath)
    cv2.imwrite(filepath, image)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
    	help="path to input image or directory of images (jpg or png)")
    parser.add_argument("-o", "--output", type=str, required=False,
        default=None,
    	help="path to output directory")
    args = vars(parser.parse_args())

    prototxt_path = 'model/colorization_deploy_v2.prototxt'
    model_path = 'model/colorization_release_v2.caffemodel'
    points_path = 'model/pts_in_hull.npy'

    # Load input images
    imgpaths = find_images(args['input'])

    print('Loading model...')
    net = load_caffe_model(prototxt_path, model_path, points_path)

    for imgpath in imgpaths:
        # Read and preprocess input
        image = read_image(imgpath)
        original_lab, resized_lab = preprocess_image(image)
        # Colorize image
        image_ab = colorize_image(net, resized_lab)
        colorized = postprocess_image(original_lab, image_ab)
        # Save colorized output
        save_image(colorized, imgpath, args['output'])
