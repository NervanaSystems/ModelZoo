##Model

This is an implementation of a deep convolutional neural network model inspired by the paper 
[Springenberg, Dosovitskiy, Brox, Riedmiller 2014](http://arxiv.org/abs/1412.6806). 

### Model script
The model run script is included in the neon repo examples [cifar10_allcnn.py](https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_allcnn.py).

### Trained weights
The trained weights file can be downloaded from AWS 
[cifar10_allcnn_e350.p][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/cifar10_allcnn/cifar10_allcnn_e350.p


### neon version
The model weight file above has been generated using neon version tag [v2.3.0]((https://github.com/NervanaSystems/neon/releases/tag/v2.3.0).
It may not work with other versions.

### Performance
This model is achieving 89.5% top-1 accuracy on the validation data set.  This accuracy is 
achieved using zca whitened, global contrast normalized data, without crops or flips.
This is the same performance we achieve running the same model configuration and data through Caffe.  


### Instructions

Download the serialized model file from the location above.  The following commands should
be run from the neon installation root directory.

To test the model performance on the validation data set use the following command:
```
python examples/cifar10_allcnn.py --model_file cifar10_allcnn_e350.p -eval 1
```

To train the model from scratch for 350 epochs, use the command:
```
python examples/cifar10_allcnn.py -b gpu -e 350 -s cifar10_allcnn_trained.p
```

## Benchmarks

Machine and GPU specs:
```
Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz
Ubuntu 14.04.2 LTS
GPU: GeForce GTX TITAN X
CUDA Driver Version 7.0
```

The run times for the fprop and bprop pass are given in the table below.  The same model configuration
is used in neon and caffe.  50 iterations are timed in each framework and only the
mean value is reported. 


```
-------------------------------------------
|    Func     | neon (mean) | caffe (mean)|
-------------------------------------------
| fprop       |    10 ms    |    19 ms    |
| bprop       |    22 ms    |    65 ms    |
| iteration   |    32 ms    |    85 ms    |
-------------------------------------------
```


## Citation

```
Jost Tobias Springenberg,  Alexey Dosovitskiy, Thomas Brox and Martin A. Riedmiller. 
Striving for Simplicity: The All Convolutional Net. 
arXiv preprint arXiv:1412.6806, 2014.
```
