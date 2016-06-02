
This is an implementation of a Fast-RCNN model inspired by the paper
[Ross Girshick 2015](http://arxiv.org/pdf/1504.08083v2.pdf) and [code](https://github.com/rbgirshick/fast-rcnn)
used for object detection.

The model in the paper and code is based on either VGG16 or Caffenet which are pre-trained using ImageNet I1K
dataset. The model presented here is based on an Alexnet pretrained in Neon using ImageNet I1K dataset. The
training script will download the Alexnet weights from neon model zoo
[Alexnet](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ImageClassification/ILSVRC2012/Alexnet)

The model from the original code processes 2 images in each minibatch and the model presented here processes minibatch
with size being multiple of 32. Both image minibatch size and how many ROIs to pull from each image are configurable
parameters in the model.

The model can be configured to train the fully-connected layers only or fine-tune the Alexnet convolution/pooling
layers during the training.

### Model script

The model run script
[fast_rcnn_alexnet.py](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ObjectLocalization/FastRCNN/fast_rcnn_alexnet.py
)
shows how to train and run this model.

### Trained weights

The trained model weights can be downloaded from AWS using the following link:
[trained Fast-RCNN model weights][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/frcn_alexnet_fc_only.p

The details of this trained model:

The model is trained using PASCAL VOC 2007 trainval set for 100 epochs.

This model is trained with only the fully connected layers. The convolution and pooling layers that were
seeded by pre-trained Alexnet stay constant during training. It is reported from the paper and also showed
in our implementation that fine-tuning fht Alexnet layers will only slightly improve the performance.
  
### Performance

This script demonstrates a training process for the Fast-RCNN model and achieves the same level of training
error (around 0.15) as the model in Caffe [code](https://github.com/rbgirshick/fast-rcnn).

### Instructions

Follow the neon [installation procedure](http://neon.nervanasys.com/docs/latest/installation.html)
before proceeding.

If neon is installed into a `virtualenv`, make sure that it is activated before running the commands below.
Also, the commands below use the GPU backend by default so add `-b cpu` if you are running on a system without
a compatible GPU. But beware that it will be very slow.

To train the model from scratch for 100 epochs, use the command:
```
python fast_rcnn_alexnet.py -s frcn_weights.p -e 100
```

To continue training the model using pre-trained weights for another 10 epochs, use the command:
```
python fast_rcnn_alexnet.py --model_file frcn_alexnet_fc_only.p -s frcn_weights.p -e 110
```

Additional options are available through `--help` option for details.
