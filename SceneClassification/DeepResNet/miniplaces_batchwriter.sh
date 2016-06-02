#!/bin/bash

### CHANGE THESE VARIABLES
SRC_DIR=/mnt/data/miniplaces  # where data.tar.gz, development_kit.tar.gz live
IMG_DIR=/scratch/minitest  # where images will be written out to
OUT_DIR=/scratch/minitest/macro  # where macrobatches will be written out to

# Unpack the images
cd $IMG_DIR
tar -xzf $SRC_DIR/data.tar.gz images --strip-components=1

# Create the label mapping files
tar -xzf $SRC_DIR/development_kit.tar.gz development_kit/data/train.txt -O | tr " " , | shuf | gzip > $IMG_DIR/train_file.csv.gz
tar -xzf $SRC_DIR/development_kit.tar.gz development_kit/data/val.txt -O | tr " " , | gzip > $IMG_DIR/val_file.csv.gz

# Write the batches
python neon/data/batch_writer.py --data_dir $OUT_DIR --image_dir $IMG_DIR --set_type csv

