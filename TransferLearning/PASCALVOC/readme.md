## Introduction

Deep neural networks have been able to acheive state of the art performance on various
canonical artificial intelligence tasks.  It is often the case that training these models
can take a long time and require a large, labeled dataset.  Once trained on one task, the
learned parameters of a model can be used to seed the layers of a new, hybrid model which
can then build upon the original model to learn a different task.  For example, the image
classification networks like Alexnet are able to extract features from an input image.
The Alexnet model can be adapted to other image classification tasks by initializing the
feature extraction parameters with those obtained by training Alexnet on the ILSVCR2012
dataset and adapting the output layers to accomadate the criterion of the new task.

The model included here is an implementation of the transfer learning paradigm described in
the [paper](http://www.di.ens.fr/willow/pdfscurrent/oquab14cvpr.pdf):
```
"Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks",
M. Oquab et. al., CVPR 2014.
```

A full description of this model implementation can be found in our
[blog](http://www.nervanasys.com/blog/).


## Useage

To start a fresh new training run that trains for 10 epochs:
```
python transfer_learning.py -e 10 -b gpu -s tlearn_save.prm --serialize 1
```
The script will download an alexnet model pretrained on the ILSVCR2012 dataset from the neon
Model Zoo to initialize the the feature extraction layers of this model.  Then training will
proceed on the PASCAL VOC dataset.

Training can take some time, make sure to serialize checkpoint using the "--serialize" option.
If training is interrupted, it can be resumed from the last saved checkpoint by adding the
"--model_file <last saved checkpoint file>" option to the training command above.

To calculate the classification errors on a saved trained model, run the same command
with the epochs set to 0 and the serialized model file as an input parameter. For
example:
```
python transfer_learning.py -e 0 -b gpu --model_file tlearn_save.prm
```

### Trained weights

The trained weights file can be downloaded from AWS using the following link:
[trained model weights][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/transfer_learning/PASCAL_VOC/tlearn_save_5000.prm


## version compatibility

These scripts have been tested with [neon library version 1.4.0](https://github.com/NervanaSystems/neon/tree/v1.4.0)
 (SHA [bc196cb](https://github.com/NervanaSystems/neon/commit/bc196cbe4131a76cd0c584e93aa7f8285b6243cb)).
