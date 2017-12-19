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
import os

import numpy as np
from neon.util.argparser import NeonArgparser
from neon.util.persist import load_obj
from neon.transforms import Misclassification, CrossEntropyMulti
from neon.optimizers import GradientDescentMomentum
from neon.layers import GeneralizedCost
from neon.models import Model

from neon.data.dataloader_transformers import OneHot, TypeCast, BGRMeanSubtract
from neon.data.aeon_shim import AeonDataLoader

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
args = parser.parse_args()

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
                    "crop_enable": True,
                    "center": True,
                    "flip_enable": False}

    return {'manifest_filename': manifest_filename,
            'manifest_root': manifest_root,
            'batch_size': batch_size,
            'subset_fraction': float(subset_pct/100.0),
            'etl': [image_config, label_config],
            'augmentation': [augmentation]}

def make_val_config(manifest_filename, manifest_root, batch_size, subset_pct=100):
    val_config = config(manifest_filename, manifest_root, batch_size, subset_pct)
    return wrap_dataloader(AeonDataLoader(val_config))

test_set = make_val_config(args.manifest["val"], args.manifest_root, batch_size=args.batch_size)

model = Model(load_obj(args.model_file))
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001)
model.initialize(test_set, cost=cost)

acc = 1.0 - model.eval(test_set, metric=Misclassification())[0]
print 'Accuracy: %.1f %% (Top-1)' % (acc*100.0)

model.benchmark(test_set, cost=cost, optimizer=opt)
