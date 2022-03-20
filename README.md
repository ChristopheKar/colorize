# Grayscale Image Colorization

This project aims to colorize grayscale images by applying the work done by Zhang et al.
in their 2016 ECCV paper, [Colorful Image Colorization](https://arxiv.org/abs/1603.08511),
which is beautifully implemented and showcased in this [repo](https://richzhang.github.io/colorization/).
This project is also partly based on this excellent
[tutorial](https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/)
by Adrian Rosebrock on [pyimagesearch](https://www.pyimagesearch.com), for the OpenCV-Caffe implementation of one of the models.

Currently, this repository implements both the ECCV16 (Caffe and PyTorch) and the SIGGRAPH17 (PyTorch) models by Zhang et al. in one simple interface, and only in inference mode. Furthermore, GPUs are not yet
supported by the code or dependencies, meaning that even if the relevant GPU drivers and libraries are installed, the inference will not yet utilize it. See [Roadmap](#Roadmap) section.

**Lebanon, Achrafieh, circa 1910 - Colorized**
![Lebanon, Achrafieh, circa 1910 - Colorized](/outputs/ashrafieh_1910.jpg)


## Quick Start
The weights for the different models can be retrieved using:
```bash
# ECCV16 Caffe model
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O weights/caffe/colorization_release_v2.caffemodel
# ECCV16 PyTorch model
wget https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth -O weights/eccv16-9b330a0b.pth
# SIGGRAPH17 PyTorch model
wget https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth -O weights/siggraph17-df00044c.pth
```

If all dependencies are installed, the example images can be colorized by running:
```bash
python colorize.py -i images/ -o outputs/
```
By default, this will use the Zhang et. al. SIGGRAPH-17 model and look for the weights under
`weights/siggraph17-df00044c.pth`.

## Installation

First, start by cloning this repository:
```bash
git clone https://github.com/ChristopheKar/colorize
```

### Dependencies

This project uses Python 3.9, and the dependencies listed in `requirements.txt` can
be installed with `pip`, `conda`, or any other package manager, in a virtual environment
or other. For example, using `pip`:
```bash
pip install -r requirements.txt
```

Currently, the dependencies do not include support for GPUs, and even if the proper

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

#### Quickstart

The easiest way to use this repository with Docker is with the provided utility script, `run.sh`.
First, make sure it is executable (`chmod +x run.sh`) and simply execute it (`./run.sh`).
By default, its entrypoint is the `colorize.py` script, so it can directly take in the arguments for that script, for example
```
# Use ./run.sh --help to show usage details
./run.sh -i images/ -o outputs -m siggraph17
```

## Roadmap

- [x] Zhang et al. ECCV16 Caffe model
- [x] Zhang et al. ECCV16 PyTorch model
- [x] Zhang et al. SIGGRAPH17 PyTorch model
- [ ] GPU Support
- [ ] Training support
- [ ] Other colorizers?

## References
- Colorful Image Colorization [[paper](https://arxiv.org/abs/1603.08511)] [[code](https://richzhang.github.io/colorization/)]
- [Black and white image colorization with OpenCV and Deep Learning, PyImageSearch](https://pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/)
- [Let there be color!](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)
