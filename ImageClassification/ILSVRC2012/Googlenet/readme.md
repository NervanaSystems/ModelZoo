## Model

This is an implementation of the GoogLeNet model for image classification described in [Szegedy et. al. 2014](http://arxiv.org/pdf/1409.4842.pdf).

The model presented here does not include any Local Response Normalization layers as were used in the published implementation.

### Model script

The model run script is included here [googlenet_neon.py](./googlenet_neon.py).


### Trained weights

The trained weights file can be downloaded from AWS using the following link:
[trained googlenet model weights][S3_WEIGHTS_FILE].

[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/googlenet/googlenet_fused_conv_bias.p

### Performance
This model is acheiving 64% top-1 and 85.5% top-5 accuracy on the validation data set.

During training, the images were randomly cropped and flipped horizontally but scale jittering and colorspace noise addition was not implemented.


### Instructions
To run the model, first the ImageNet data set needs to be uploaded and converted to the format compatible with neon (see  [instructions](http://neon.nervanasys.com/docs/latest/datasets.html#imagenet)).  Note there has been some changes to the format of the mean data subtraction; users with the old format may be prompted to run an update script before proceeding.


This script works with the [neon release v2.3.0](https://github.com/NervanaSystems/neon/tree/v2.3.0).  Make sure that your local repo is synced to this commit and run the [installation procedure](http://neon.nervanasys.com/docs/latest/installation.html) before proceeding.


If neon is installed into a `virtualenv`, make sure that it is activated before running the commands below.  Also, the commands below use the GPU backend by default so add `-b cpu` if you are running on a system without a compatible GPU.


To test the model performance on the validation data set and benchmark the run times use the following command:
```
python googlenet_neon.py -w path/to/dataset/batches --model_file googlenet.p
```

Additional options are available to add features like saving checkpoints and displaying logging information, use the `--help` option for details.  For information on generating the ILSVRC2012 data ste macrobacthes check out the
neon documentation [page](http://neon.nervanasys.com/docs/latest/datasets.html#imagenet).

## Training
Training this model requires some features to neon which will be released soon.  These scripts will be updated to include the training procedure as soon as possible.

## Benchmarks

Machine and GPU specs:
```
Intel(R) Core(TM) i5-4690 CPU @ 3.50GHz
Ubunutu 14.04
GPU: GeForce GTX TITAN X
CUDA Driver Version 7.0
```

The run times for the fprop and bprop pass and the parameter update are given in the table below.  The iteration row is the combined runtime for all functions in a training iteration.  These results are for each minibatch consisting of 128 images of shape 224x224x3.  The model was run 12 times, the first two passes were ignored and the last 10 were used to get the benchmark results.
```
------------------------------
|    Func     |      Mean    |
------------------------------
| fprop       |   116 msec   |
| bprop       |   261 msec   |
| update      |    45 msec   |
| iteration   |   424 msec   |
------------------------------
```


## Citation

```
Going deeper with convolutions
Szegedy, Christian; Liu, Wei; Jia, Yangqing; Sermanet, Pierre; Reed, Scott; Anguelov, Dragomir;
Erhan, Dumitru; Vanhoucke, Vincent; Rabinovich, Andrew
arXiv:1409.4842
```
