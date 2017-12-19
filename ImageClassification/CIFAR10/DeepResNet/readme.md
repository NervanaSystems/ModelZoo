# Model
This is an implementation of the deep residual network used for cifar10 as described in [He et. al.,
"Deep Residual Learning for Image Recognition"](http://arxiv.org/abs/1512.03385).  The model is
structured as a very deep network with skip connections designed to have convolutional parameters
adjusting to residual activations.  The training protocol uses minimal pre-processing (mean
subtraction) and very simple data augmentation (shuffling, flipping, and cropping).  All model
parameters (even batch norm parameters) are updated using simple stochastic gradient descent with
weight decay.  The learning rate is dropped only twice (at 90 and 123 epochs).

### Acknowledgments
Many thanks to Dr. He and his team at MSRA for their helpful input in replicating the model as
described in their paper.

### Model script
The model train script is included ([resnet_cifar10.py](./resnet_cifar10.py)).

### Trained weights
The trained weights file for a model with depth of 56 can be downloaded from AWS
[resnet56_e179.p][S3_WEIGHTS_FILE]
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/resnet/cifar10/resnet56_e179.p

### Performance
Training this model with the options described below should be able to achieve above 92.7% top-1
accuracy using only mean subtraction, random cropping, and random flips.

## Instructions
This script was tested with [neon version 2.2.0](https://github.com/NervanaSystems/neon/tree/v2.2.0).
Make sure that your local repo is synced to this commit and run the [installation
procedure](http://neon.nervanasys.com/docs/latest/installation.html) before proceeding.
Commit SHA for v2.2.0 is  `5843e7116d880dfc59c8fb558beb58dd2ef421d0`

This example uses the `DataLoader` module to load the images for consumption while applying random
cropping, flipping, and shuffling.  To use the DataLoader, the script will generate PNG files from
the CIFAR-10 dataset the first time it is run and will saved them to the directory provided by
the "-w" or "--data_dir" option to the script.
Note that it is good practice to choose your `data_dir` to be local to your machine in order to
avoid having `DataLoader` module perform reads over the network.

Once the batches have been written out, you may initiate training:
```
resnet_cifar10.py -r 0 -vv \
    --log <logfile> \
    --epochs 180 \
    --save_path <model-save-path> \
    --eval_freq 1 \
    --backend gpu \
    --data_dir <path-to-saved-batches> \
    --depth <n>
```

The depth argument is the `n` value discussed in the paper which represents the number of repeated
residual models at each filter depth.  Since there are 3 stages at each filter depth, and each
residual module consists of 2 convolutional layers, there will be `6n` total convolutional layers
in the residual part of the network, plus 2 additional layers (input convolutional, and output
linear), making the total network `6n+2` layers deep.  For depth arguments of 3, 5, 9, 18, we get
network depths of 20, 32, 56, and 110.

If you just want to run evaluation, you can use the much simpler script that loads the serialized
model and evaluates it on the validation set and runs a speed benchmark:

```
resnet_eval.py -vv --model_file <model-save-path> -w <path_to_dataset>
```

## Benchmarks
Machine and GPU specs:
```
Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz
Ubuntu 14.04.2 LTS
GPU: GeForce GTX TITAN X
CUDA Driver Version 7.0
```

For a batch size of 128 images, the runtimes are:
```
-----------------------------
|    Func     |    Mean     |
-----------------------------
| fprop       |   21.0  ms  |
| bprop       |   82.3  ms  |
 ------------- -------------
| Total       |  103.3  ms  |
-----------------------------
```
