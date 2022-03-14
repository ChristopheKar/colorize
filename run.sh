#!/bin/sh

default_name="colorize"
default_img="colorize"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) help=1 ;;
        -s|--shell) shell=1 ;;
        -b|--build) build=1 ;;
        -r|--as-root) root=1 ;;
        -n|--name) name="$2"; shift ;;
        -in|--img-name) imgname="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
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

# Build image
if [[ "$build" -eq 1 ]]; then
    echo "Building image..."
    docker build -t $imgname .
fi

# Set entrypoint
if [[ "$shell" -eq 1 ]]; then
    entrypoint="--entrypoint /bin/bash"
fi

# Set container user id
if [[ "$root" -eq 1 ]]; then
    uid=0
    gid=0
else
    uid=$(id -u)
    gid=$(id -g)
fi

# Run container
docker run \
    -it --rm \
    --user $uid:$gid \
    --name $name \
    -v $PWD:/app \
    $entrypoint \
    $imgname