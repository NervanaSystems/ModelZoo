#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation
from neon.layers import MergeSum, SkipNode
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks

import os

from neon.data.dataloader_transformers import OneHot, TypeCast, BGRMeanSubtract
from neon.data.aeon_shim import AeonDataLoader

def wrap_dataloader(dl, dtype=np.float32):
    dl = OneHot(dl, index=1, nclasses=10)
    dl = TypeCast(dl, index=0, dtype=dtype)
    dl = BGRMeanSubtract(dl, index=0)
    return dl

def config(manifest_filename, manifest_root, batch_size, subset_pct):
    image_config = {"type": "image",
                    "height": 32,
                    "width": 32}
    label_config = {"type": "label",
                    "binary": False}
    augmentation = {"type": "image",
                    "crop_enable": True}

    return {'manifest_filename': manifest_filename,
            'manifest_root': manifest_root,
            'batch_size': batch_size,
            'subset_fraction': float(subset_pct/100.0),
            'etl': [image_config, label_config],
            'augmentation': [augmentation]}


def make_train_config(manifest_filename, manifest_root, batch_size, subset_pct=100):
    train_config = config(manifest_filename, manifest_root, batch_size, subset_pct)
    train_config['augmentation'][0]['center'] = False
    train_config['augmentation'][0]['flip_enable'] = True
    train_config['shuffle_enable'] = True
    train_config['shuffle_manifest'] = True

    return wrap_dataloader(AeonDataLoader(train_config))


def make_val_config(manifest_filename, manifest_root, batch_size, subset_pct=100):
    val_config = config(manifest_filename, manifest_root, batch_size, subset_pct)
    return wrap_dataloader(AeonDataLoader(val_config))

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 6n+2)')
args = parser.parse_args()

# setup data provider
train = make_train_config(args.manifest['train'], args.manifest_root, args.batch_size)
test = make_val_config(args.manifest['val'], args.manifest_root, args.batch_size)

def conv_params(fsize, nfm, stride=1, relu=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=True)


def id_params(nfm):
    return dict(fshape=(1, 1, nfm), strides=2, padding=0, activation=None, init=IdentityInit())


def module_factory(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**id_params(nfm))]
    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module

# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
nfms = [2**(stage + 4) for stage in sorted(range(3) * args.depth)]
strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

# Now construct the network
layers = [Conv(**conv_params(3, 16))]
for nfm, stride in zip(nfms, strides):
    layers.append(module_factory(nfm, stride))
layers.append(Pooling(8, op='avg'))
layers.append(Affine(nout=10, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

model = Model(layers=layers)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001,
                              schedule=Schedule([90, 123], 0.1))

# configure callbacks
callbacks = Callbacks(model, eval_set=test, metric=Misclassification(), **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
