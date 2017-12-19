##Model

Here we have ported the weights for the 16 and 19 layer VGG models from the Caffe model zoo (see [link](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014))

### Model script
The model run script is included here [vgg_neon.py](./vgg_neon.py).  This script can easily be adapted for fine tuning this network but we have focused on inference here because a successful training protocol may require details beyond what is available from the Caffe model zoo.

### Trained weights
The trained weights file can be downloaded from AWS using the following links:
[VGG_D.p]( https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/VGG_D_fused_conv_bias.p) and [VGG_E.p][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/VGG_E_fused_conv_bias.p

## Performance

### Accuracy

Testing the image classification performance for the two models on the ILSVRC 2012 validation data set gives the results in the table below:
```
 ------------------------------
|         |       Accuracy     |
| Model   |  Top 1   |  Top 5  |
 ------------------------------
| VGG D   |  69.2 %  | 88.9 %  |
| VGG E   |  69.3 %  | 88.8 %  |
 ------------------------------
```

These results are calculated using a single scale, using a 224x224 crop of each image.  These results are comparable to the classification accuracy we computed using the Caffe model zoo 16 and 19 layer VGG models using Caffe [Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014).




### Speed

We ran speed benchmarks on this model using neon.  These results are using a 64 image batch size with 3x224x224 input images.  The results are in the tables below:

#### VGG D
```
 ----------------------
|    Func  |    Time   |
 ----------------------
| fprop    |   366 ms  |
| bprop    |   767 ms  |
| update   |    19 ms  |
 ----------------------
| Total    |  1152 ms  |
 ----------------------
```

#### VGG E
```
 -----------------------
|    Func  |    Time    |
 -----------------------
| fprop    |    452 ms  |
| bprop    |    940 ms  | 
| update   |     20 ms  |
 -----------------------
| Total    |   1412 ms  |
 -----------------------
```
The run times for the fprop and bprop pass and the parameter update are given in the table below. The iteration row is the combined runtime for all functions in a training iteration. These results are for each minibatch consisting of 64 images of shape 224x224x3. The model was run 12 times, the first two passes were ignored and the last 10 were used to get the benchmark results.



System specs:
```
Intel(R) Core(TM) i5-4690 CPU @ 3.50GHz
Ubunutu 14.04
GPU: GeForce GTX TITAN X
CUDA Driver Version 7.0
```

## Instructions

Make sure that your local repo is synced to the proper neon repo commit (see version below) and run the [installation procedure](http://neon.nervanasys.com/docs/latest/installation.html) before proceeding.  To run
this model script on the ILSVRC2012 dataset, you will need to have the data in the neon macrobatch format; follow
the instructions in the neon documentations for [setting up the data sets](http://neon.nervanasys.com/docs/latest/datasets.html#imagenet).

If neon is installed into a `virtualenv`, make sure that it is activated before running the commands below.

To run the evaluation of the model:
```
# for 16 layer VGG D model
python vgg_neon.py --vgg_ver D --model_file VGG_D.p -w path/to/dataset/batches -z 64 --caffe

# for 16 layer VGG D model
python vgg_neon.py --vgg_ver E --model_file VGG_E.p -w path/to/dataset/batches -z 64 --caffe
```

Note that the `--caffe` option is needed to match the dropout implementation used by Caffe.

The batch size is set to 64 in the examples above because with larger batch size the model may not fit on some GPUs.  Use smaller batch sizes if necessary.  The script given here can easily be altered for model fine tuning.  See the neon user manual for help with that.


### Version compatibility

Neon version: v2.3.0 (https://github.com/NervanaSystems/neon/tree/v2.3.0).

## Citation

```
Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556
```

## License

For the model weight files please abide by the license posted with the Caffe weights files:
[Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).

