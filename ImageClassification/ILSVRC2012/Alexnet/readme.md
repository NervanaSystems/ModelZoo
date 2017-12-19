##Model

This is an implementation of a deep convolutional neural network model inspired by
the paper
[Krizhevsky,Sutskever, Hinton 2012](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
used to classify images from the ImageNet 2012 competition.  The model used here is
adapted from the Caffenet architecture described
[here](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel).

The neon implementation does not use the conv layer weight grouping as used in the
Caffe model.

### Model script
The model run script is included below [alexnet_neon.py](./alexnet_neon.py).

### Trained weights
The trained weights file can be downloaded from AWS using the following link:
[trained Alexnet model weights][S3_WEIGHTS_FILE]
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/alexnet/alexnet_fused_conv_bias.p

### Performance
This model is acheiving 58.6% top-1 and 81.1% top-5 accuracy on the validation
data set after 90 epochs of training.  The training here is using a single, random crop on every epoch and
flipping the images across the vertical axis.  These results improve further
with additional data augmentation added to the training as decsribed in
[Krizhevsky,Sutskever, Hinton 2012](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

### Instructions
To run the model, first the ImageNet data set needs to be uploaded and
converted to the format compatible with neon (see
[instructions](http://neon.nervanasys.com/docs/latest/datasets.html#imagenet)).
Note there has been some changes to the format of the mean data subtraction;
users with the old format may be prompted to run an update script before proceeding.


This script was tested with the [neon release v2.3.0](https://github.com/NervanaSystems/neon/tree/v2.3.0).
Make sure that your local repo is synced to this commit and run the
[installation procedure](http://neon.nervanasys.com/docs/latest/installation.html)
before proceeding.


If neon is installed into a `virtualenv`, make sure that it is activated before
running the commands below.  Also, the commands below use the GPU backend by default
so add `-b cpu` if you are running on a system without a compatible GPU.


To test the model performance on the validation data set use the following command:
```
python alexnet_neon.py -w path/to/dataset/batches --model_file alexnet.p --test_only
```

To train the model from scratch for 90 epochs, use the command:
```
python alexnet_neon.py -w path/to/dataset/batches -s alexnet_weights.p -e 90
```

Additional options are available to add features like saving checkpoints
and displaying logging information, use the `--help` option for details.


## Benchmarks

Machine and GPU specs:
```
Intel(R) Core(TM) i5-4690 CPU @ 3.50GHz
Ubunutu 14.04
GPU: GeForce GTX TITAN X
CUDA Driver Version 7.0
```

The run times for the fprop and bprop pass and the parameter update are given
in the table below.  The iteration row is the combined runtime for all functions
in a training iteration.  These results are for each minibatch consisting of 128
images of shape 224x224x3.  The model was run 12 times, the first two passes were
ignored and the last 10 were used to get the benchmark results.
```
------------------------------
|    Func     |      Mean    |
------------------------------
| fprop       |   40.1 msec  |
| bprop       |   83.1 msec  |
| iteration   |  123.2 msec  |
------------------------------
```


## Citation

```
ImageNet Classification with Deep Convolutional Neural Networks
Alex Krizhevsky and Sutskever, Ilya and Geoffrey E. Hinton
Advances in Neural Information Processing Systems 25
eds.F. Pereira and C.J.C. Burges and L. Bottou and K.Q. Weinberger
pp. 1097-1105, 2012
```

