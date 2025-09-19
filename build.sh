#!/bin/bash
export DOCKER_BUILDKIT=0 

echo "Building base images..."
docker build -t base-python -f Dockerfile.base-python .
docker build -t pytorch-base -f Dockerfile.pytorch-base .
docker build -t cv-base -f Dockerfile.cv-base .
docker build -t ocr-base -f Dockerfile.ocr-base .
docker build -t ffmpeg-base -f Dockerfile.ffmpeg-base .

echo "Tagging images for compose..."
docker tag base-python localhost/base-python
docker tag pytorch-base localhost/pytorch-base
docker tag cv-base localhost/cv-base
docker tag ocr-base localhost/ocr-base
docker tag ffmpeg-base localhost/ffmpeg-base

echo "Building services..."
docker-compose -f docker-compose-test.yml build

echo "Done!"
