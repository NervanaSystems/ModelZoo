#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Simplified version of VGG model D and E
Based on manuscript:
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556
"""
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Constant, GlorotUniform, Xavier
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks

from neon.data.dataloader_transformers import OneHot, TypeCast, BGRMeanSubtract
from neon.data.aeon_shim import AeonDataLoader

def wrap_dataloader(dl, dtype=np.float32):
    dl = OneHot(dl, index=1, nclasses=1000)
    dl = TypeCast(dl, index=0, dtype=dtype)
    dl = BGRMeanSubtract(dl, index=0)
    return dl

def common_config(subset_pct, manifest_filename, manifest_root, batch_size, scale):
    image_config = {"type": "image",
                    "height": 224,
                    "width": 224}
    label_config = {"type": "label",
                    "binary": False}
    augmentation = {"type": "image",
                    "scale": scale,
                    "crop_enable": True}

    return {'manifest_filename': manifest_filename,
            'manifest_root': manifest_root,
            'batch_size': batch_size,
            'subset_fraction': float(subset_pct/100.0),
            'etl': [image_config, label_config],
            'augmentation': [augmentation]}

def make_train_config(subset_pct, manifest_filename, manifest_root, batch_size, scale):
    train_config = common_config(subset_pct, manifest_filename, manifest_root, batch_size, scale)
    train_config['shuffle_enable'] = True
    train_config['shuffle_manifest'] = True
    return wrap_dataloader(AeonDataLoader(train_config))

def make_val_config(subset_pct, manifest_filename, manifest_root, batch_size, scale):
    val_config = common_config(subset_pct, manifest_filename, manifest_root, batch_size, scale)
    return wrap_dataloader(AeonDataLoader(val_config))

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--vgg_version', default='D', choices=['D', 'E'],
                    help='vgg model type')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
parser.add_argument('--test_only', action='store_true',
                    help='skip fitting - evaluate metrics on trained model weights')
args = parser.parse_args()

# Training set has different scale configuration (256, 384) than testing set (256, 256). Why?
# Commenting it out gives the folowing runtime error:
# RuntimeError: Unable to create internal loader object: value for 'scale' out of range
#train = make_train_config(args.subset_pct, args.manifest["train"], args.manifest_root, batch_size=args.batch_size, scale=[0.875, 0.583])
val = make_val_config(args.subset_pct, args.manifest["val"], args.manifest_root, batch_size=args.batch_size, scale=[0.875, 0.875])

init1 = Xavier(local=True)
initfc = GlorotUniform()

relu = Rectlin()
conv_params = {'init': init1,
               'strides': 1,
               'padding': 1,
               'bias': Constant(0),
               'activation': relu}

# Set up the model layers
layers = []

# set up 3x3 conv stacks with different feature map sizes
for nofm in [64, 128, 256, 512, 512]:
    layers.append(Conv((3, 3, nofm), **conv_params))
    layers.append(Conv((3, 3, nofm), **conv_params))
    if nofm > 128:
        layers.append(Conv((3, 3, nofm), **conv_params))
        if args.vgg_version == 'E':
            layers.append(Conv((3, 3, nofm), **conv_params))
    layers.append(Pooling(2, strides=2))

layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=4096, init=initfc, bias=Constant(0), activation=relu))
layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=1000, init=initfc, bias=Constant(0), activation=Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model = Model(layers=layers)

# configure callbacks
top5 = TopKMisclassification(k=5)
callbacks = Callbacks(model, eval_set=val, metric=top5, **args.callback_args)

model.load_params(args.model_file, load_states=False)
mets=model.eval(val, metric=TopKMisclassification(k=5))
print 'Validation set metrics:'
print 'LogLoss: %.2f, Accuracy: %.1f %% (Top-1), %.1f %% (Top-5)' % (mets[0],
                                                                    (1.0-mets[1])*100,
                                                                    (1.0-mets[2])*100)
