# Grayscale Image Colorization

This project aims to colorize grayscale images by applying the work done by Zhang et al.
in their 2016 ECCV paper, [Colorful Image Colorization](https://arxiv.org/abs/1603.08511),
which is beautifully implemented and showcased in this [repo](https://richzhang.github.io/colorization/).

Currently, this implementation is based on the old Caffe implementation available
[here](https://github.com/richzhang/colorization/tree/caffe), so the model files can
be retrieved from there.

**Lebanon, Jounieh, circa 1894 - Colorized**
![Lebanon, Jounieh, circa 1894 - Colorized](/outputs/jounieh_1894.jpg)


## Quick Start
Model files are provided in `model/`, but the main weights file can be retrieved using:
```bash
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ./models/colorization_release_v2.caffemodel
```

If all dependencies are installed, the example images can be colorized by running:
```bash
python colorize_image.py -i images/ -o outputs/
```

## Installation

First, start by cloning this repository:
```bash
git clone https://github.com/ChristopheKar/colorize
```

### Dependencies

This project uses Python 3, and the dependencies listed in `requirements.txt` can
be installed with `pip`, `conda`, or any other package manager, in a virtual environment
or other. For example, using `pip`:
```bash
pip install -r requirements.txt
```

### Docker

The environment can also be setup using Docker and the provided `Dockerfile`.
First, build the image by running the following command in this repository:
```bash
docker build -t colorize .
```

Then, using the built image is as simple as:
```bash
docker run -it --rm --name colorize -v $PWD:/app colorize bash
```

This will drop you in an interactive bash session inside the Docker container.\
The `-v` option allows you to mount your current workspace to `/app`
inside the container, so that your files are accessible from there, and so that any
changes made to files under that path persists on your local storage. Any other changes
made inside the Docker container, e.g. installing additional packages or creating files
outside of `/app`, will not persist across sessions unless you commit your changes
to the Docker image.

## Future Steps

As previously stated, this is an old Caffe implementation. Zhang et al.'s work has been
redesigned in PyTorch, so the natural next step is to implement that newer version.
