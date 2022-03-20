#!/bin/bash

default_name="colorize"
default_img="colorize"
default_context="."

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) help=1 ;;
        -s|--shell) shell=1 ;;
        -b|--build) build=1 ;;
        -r|--as-root) root=1 ;;
        -g|--gpus) gpu_enabled=1; gpus="$2"; shift ;;
        -n|--name) name="$2"; shift ;;
        -in|--img-name) imgname="$2"; shift ;;
        -c|--build-context) context="$2"; shift ;;
        --base-image) base_image="$2"; shift ;;
        *) args="$@"; break ;;
    esac
    shift
done

# Show help and exit
if [[ "$help" -eq 1 ]]; then
    echo "usage: run.sh [-h/--help] [-s/--shell] [-b/--build] [-r/--as-root] [-n/--name NAME] [-in/--img-name IMAGENAME]"
    echo "  -h/--help: show this help message and exit"
    echo "  -s/--shell: run container with shell as entrypoint"
    echo "  -b/--build: build image before running container"
    echo "  -n/--name NAME: specify container name, default is $default_name"
    echo "  -in/--img-name IMAGENAME: specify image name, default is $default_img"
    echo "  -r/--as-root: run container as root with uid 0 and gid 0, default is current user uid and gid"
    echo "  -c/--build-context: context for building docker image, default is current directory"
    echo "  -g/--gpus: gpu devices to map to Docker container, also used to determine Docker image"
    echo "  --base-image: base image to use to build Docker image, overrides default which depends on -g/--gpus being specified"
    echo "  All other arguments are passed to the container entrypoint if it is not the shell"
    exit 0
fi

# Set container name
if [ -z ${name+x} ]; then
    name=$default_name
fi

# Set image name
if [ -z ${imgname+x} ]; then
    imgname=$default_img
fi

# Build image if --build is specified or if it does not exist locally
# Inspect image and check if exit code is 0
docker inspect "$imgname" > /dev/null 2>&1
status="$?"
if [[ "$build" -eq 1 ]] || [[ "$status" != 0 ]]; then
    echo "Building image..."

    # Set build context
    if [ -z ${context+x} ]; then
        context=$default_context
    fi

    # Set base image based on gpu
    if [[ "$gpu_enabled" -eq 1 ]]; then
      build_args="--build-arg BASE_IMAGE=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel"
    else
      build_args="--build-arg BASE_IMAGE=python:3.9-slim-buster"
    fi

    # Override default base image if specified
    if [ ! -z ${base_image+x} ]; then
        build_args="--build-arg BASE_IMAGE="$base_image""
    fi

    echo "$build_args"

    # Build image
    docker build -t $imgname $build_args "$context"
fi

# Set entrypoint as shell
if [[ "$shell" -eq 1 ]]; then
    entrypoint="--entrypoint /bin/bash"
    args=""
fi

# Set container user id
if [[ "$root" -eq 1 ]]; then
    uid=0
    gid=0
else
    uid=$(id -u)
    gid=$(id -g)
fi

# Set gpu args
if [[ "$gpu_enabled" -eq 1 ]]; then
  if [ -z $gpus ]; then
      gpu_args="--gpus all"
  else
      gpu_args="--gpus "$gpus""
  fi

else
  gpu_args=""
fi

echo "$gpu_args"

# Run container
docker run \
    -it --rm \
    --user $uid:$gid \
    --name $name \
    -v $PWD:/app \
    $gpu_args \
    $entrypoint \
    $imgname \
    $args
