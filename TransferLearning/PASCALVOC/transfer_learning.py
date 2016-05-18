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
Implementation of this paper: 
Oquab, M. and Bottou, L. and Laptev, I. and Sivic, J., Learning 
and Transferring Mid-Level Image Representations using Convolutional 
Neural Networks. CVPR 2014.

"""
import os
import math
import numpy as np
from PIL import Image

from neon.backends import gen_backend
from pascal_voc import (PASCALVOCTrain, PASCAL_VOC_CLASSES, FRCN_PIXEL_MEANS, 
                                  FRCN_IMG_DIM_SWAP, PASCAL_VOC_NUM_CLASSES)
from neon.data.datasets import Dataset
from neon.initializers import Gaussian, Constant
from neon.transforms import (Rectlin, Softmax, Identity, CrossEntropyMulti,
                             Misclassification)
from neon.transforms.cost import Metric
from neon.models import Model
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.layers import (Conv, Pooling, Affine, Dropout, RoiPooling,
                         BranchNode, Multicost, GeneralizedCost,
                         GeneralizedCostMask, Tree)
from neon.callbacks.callbacks import Callbacks, Callback
from neon.util.persist import load_obj

DEBUG = True
PASCAL_VOC_BACKGROUND_CLASS = 0

def debug(str):
    if DEBUG:
        print str

def main():
    # Collect the user arguments and hyper parameters
    args, hyper_params = get_args_and_hyperparameters()
    
    np.set_printoptions( precision=8, suppress=True, edgeitems=6, threshold=2048)
    
    # setup the CPU or GPU backend
    be = gen_backend(**extract_valid_args(args, gen_backend))
    
    # load the training dataset. This will download the dataset from the web and cache it
    # locally for subsequent use.
    train_set = MultiscaleSampler('trainval', '2007', samples_per_img=hyper_params.samples_per_img, 
                                 sample_height=224, path=args.data_dir, 
                                 samples_per_batch=hyper_params.samples_per_batch,
                                 max_imgs = hyper_params.max_train_imgs,
                                 shuffle = hyper_params.shuffle)

    # create the model by replacing the classification layer of AlexNet with 
    # new adaptation layers
    model, opt = create_model( args, hyper_params)
    
    # Seed the Alexnet conv layers with pre-trained weights
    if args.model_file is None and hyper_params.use_pre_trained_weights:
        load_imagenet_weights(model, args.data_dir)
    
    train( args, hyper_params, model, opt, train_set)
    
    # Load the test dataset. This will download the dataset from the web and cache it
    # locally for subsequent use.
    test_set = MultiscaleSampler('test', '2007', samples_per_img=hyper_params.samples_per_img, 
                                 sample_height=224, path=args.data_dir, 
                                 samples_per_batch=hyper_params.samples_per_batch,
                                 max_imgs = hyper_params.max_test_imgs,
                                 shuffle = hyper_params.shuffle)
    test( args, hyper_params, model, test_set)
    
    return

# parse the command line arguments
def get_args_and_hyperparameters():
    parser = NeonArgparser(__doc__)
    args = parser.parse_args(gen_be=False)
    
    # Override save path if None
    if args.save_path is None:
        args.save_path = 'frcn_alexnet.pickle'
    
    if args.callback_args['save_path'] is None:
        args.callback_args['save_path'] = args.save_path
    
    if args.callback_args['serialize'] is None:
        args.callback_args['serialize'] = min(args.epochs, 10)
    
    
    # hyperparameters
    args.batch_size = 64
    hyper_params = lambda: None
    hyper_params.use_pre_trained_weights = True # If true, load pre-trained weights to the model
    hyper_params.max_train_imgs = 5000 # Make this smaller in small trial runs to save time
    hyper_params.max_test_imgs = 5000 # Make this smaller in small trial runs to save time
    hyper_params.num_epochs = args.epochs
    hyper_params.samples_per_batch = args.batch_size # The mini-batch size
    # The number of multi-scale samples to make for each input image. These
    # samples are then fed into the network in multiple minibatches.
    hyper_params.samples_per_img = hyper_params.samples_per_batch*7 
    hyper_params.frcn_fine_tune = False
    hyper_params.shuffle = True
    if hyper_params.use_pre_trained_weights:
        # This will typically train in 10-15 epochs. Use a small learning rate
        # and quickly reduce every 5-10 epochs. Use a high momentum since we
        # are close to the minima.
        s = 1e-4
        hyper_params.learning_rate_scale = s
        hyper_params.learning_rate_sched = Schedule(step_config=[15, 20], 
                                        change=[0.1*s, 0.01*s])
        hyper_params.momentum = 0.9
    else: # need to be less aggressive with reducing learning rate if the model is not pre-trained
        s = 1e-2
        hyper_params.learning_rate_scale = 1e-2
        hyper_params.learning_rate_sched = Schedule(step_config=[8, 14, 18, 20], 
                                        change=[0.5*s, 0.1*s, 0.05*s, 0.01*s])
        hyper_params.momentum = 0.1
    hyper_params.class_score_threshold = 0.000001
    hyper_params.score_exponent = 5
    hyper_params.shuffle = True
    return args, hyper_params

def create_model(args, hyper_params):
    # setup layers
    imagenet_layers = [
        Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(),
             padding=3, strides=4),
        Pooling(3, strides=2),
        Conv((5, 5, 192), init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin(),
             padding=2),
        Pooling(3, strides=2),
        Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(),
             padding=1),
        Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
             padding=1),
        Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
             padding=1),
        Pooling(3, strides=2),
        Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
        # The following layers are used in Alexnet, but are not used in the new model
        Dropout(keep=0.5),
        # Affine(nout=1000, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax())
    ]
    
    target_layers = imagenet_layers + [    
        Affine(nout=4096, init=Gaussian(scale=0.005), bias=Constant(.1), activation=Rectlin()),
        Dropout(keep=0.5),
        Affine(nout=21, init=Gaussian(scale=0.01), bias=Constant(0), activation=Softmax())]
    
    # setup optimizer
    opt = GradientDescentMomentum(hyper_params.learning_rate_scale, 
                                  hyper_params.momentum, wdecay=0.0005,
                                  schedule=hyper_params.learning_rate_sched)
    
    # setup model
    if args.model_file:
        model = Model(layers=args.model_file)
    else:
        model = Model(layers=target_layers)
    
    return model, opt

def load_pre_trained_weight(i, layer):
    return layer.name != 'Linear_2'

def load_imagenet_weights(model, path):
    # load a pre-trained Alexnet from Neon model zoo to the local
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/alexnet/old/pre_v1.4.0/'
    filename = 'alexnet.p'
    size = 488808400

    workdir, filepath = Dataset._valid_path_append(path, '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)

    print 'Loading the Alexnet pre-trained with ImageNet I1K from: ' + filepath
    pdict = load_obj(filepath)

    param_layers = [l for l in model.layers.layers]
        
    param_dict_list = pdict['model']['config']['layers']
    skip_loading = False
    for i, layer in enumerate(param_layers):
        if not load_pre_trained_weight(i, layer):
            skip_loading = True
        if not skip_loading:
            ps = param_dict_list[i]
            print "Loading weights for:{} [src: {}]".format(layer.name, ps['config']['name'])
            layer.load_weights(ps, load_states=True)
        else:
            config_name = param_dict_list[i]['config']['name'] if i < len(param_dict_list) else ""
            print "Skipped loading weights for: {} [src: {}]".format(layer.name, config_name)
        
    return


class EpochEndCallback(Callback):
    def on_epoch_end(self, callback_data, model, epoch):
        print "Epoch {} avg cost: {}".format(epoch, model.total_cost.get())

def train(args, hyper_params, model, opt, data_set):
    # setup cost function as CrossEntropy
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    
    callbacks = Callbacks(model, **args.callback_args)
    callbacks.add_callback(EpochEndCallback())
    
    data_set.set_mode('train')
    model.fit(data_set, optimizer=opt,
              num_epochs=hyper_params.num_epochs, cost=cost, callbacks=callbacks)
    
    return

def test( args, hyper_params, model, data_set):
    data_set.set_mode('test')
    print 'Running inference on the test dataset...'
    metric = ImageScores( args.backend, PASCAL_VOC_NUM_CLASSES, data_set.num_imgs,
                          hyper_params.samples_per_batch, data_set.batches_per_img,
                          hyper_params.score_exponent)
    misclassification_error = model.eval(data_set, metric=metric)
    debug('Misclassification error (on test dataset): {}'.format( misclassification_error*100.0))
    
    print 'Evaluating the results of the inference run on the test dataset...'
    metric.evaluation( data_set.roi_db, data_set.shuf_idx, 
                       hyper_params.class_score_threshold)
    
    return
    
class ImagePatch(object):
    def __init__(self, id, img, location, size, scale):
        self.id = id
        self.img = img
        self.location = location
        self.size = size
        self.scale = scale
        size_half = size * 0.5
        self.bbox = ( location[0] - size_half, location[1] - size_half,
                      location[0] + size_half, location[1] + size_half)
        self.area = size*size
        self.label = -1
        self.matched_roi_idx = -1
        
    def __str__(self):
        return "{}".format(self.bbox)
        
    def overlap_area(self, bbox):
        area = max(0, min(self.bbox[2], bbox[2]) - max(self.bbox[0], bbox[0])) *\
               max(0, min(self.bbox[3], bbox[3]) - max(self.bbox[1], bbox[1]))
        return float(area)

class PascalImage(object):
    def __init__(self, img_file_path, rois, labels):
        assert rois.shape[0] == labels.shape[0]
        self.img_file_path = img_file_path
        self.pil = Image.open( img_file_path)  # This is RGB order
        self.shape = (self.pil.size[0], self.pil.size[1])
        self.num_rois = len(rois)
        self.rois = rois # bounding box (x0, y0, x1, y1)
        self.labels = labels
        self.background_patches = []
        self.non_background_patches = []
        
        dir, self.file_name = os.path.split(img_file_path)
        debug_dir = dir + '/../debug/'
        self.debug_file_path_prefix = debug_dir + self.file_name[:-4] # remove the '.jpg'
        
        if DEBUG:
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
        
        return
    
    def sample_locs_for_dim(self, width, size, num_samples):
        assert size <= width and num_samples > 0
        w = width-1.0
        if num_samples == 1:
            return [w/2]
        
        minmax = (size/2.0, w-size/2.0)
        step_size = (minmax[1] - minmax[0]) / (num_samples - 1)
        samples = [minmax[0] + i*step_size for i in range(num_samples)]
        return samples
    
    def get_label_for_patch(self, patch):
        chosen_idx = -1
        for idx in range(self.num_rois):
            roi_bbox = [float(i) for i in self.rois[idx]]
            roi_area = float(roi_bbox[2]-roi_bbox[0]) * float(roi_bbox[3]-roi_bbox[1])
            overlap_area = patch.overlap_area(roi_bbox)
            if overlap_area >= 0.2 * patch.area and \
               overlap_area >= 0.6 * roi_area:
                if chosen_idx == -1:
                    chosen_idx = idx
                else: # it already met this criteria with another class
                    return PASCAL_VOC_BACKGROUND_CLASS, -1

        label = self.labels[chosen_idx] if chosen_idx != -1 else PASCAL_VOC_BACKGROUND_CLASS
        if label != PASCAL_VOC_BACKGROUND_CLASS:
            debug("P_id:{} P_bbox:{} ROI:{} bbox:{} label:{}".format( patch.id, patch.bbox, 
                       chosen_idx, self.rois[chosen_idx], label))

        return label, chosen_idx
        
    def compute_patches_at_scale(self, scale_idx, scale, p_id_base):
        debug("Processing {} scale_idx:{} scale:{}".format(self.file_name, scale_idx, scale))
        shape = np.array(self.shape)
        size = (np.amin(shape)-1) / scale
        num_samples = np.ceil( (shape-1) / size)
        num_samples = [int(n*2) if n > 1 else int(n) for n in num_samples]
        patches = []
        sample_locs = [ self.sample_locs_for_dim( self.shape[0], size, num_samples[0]),
                        self.sample_locs_for_dim( self.shape[1], size, num_samples[1])]
        p_id = p_id_base
        for sample_loc_0 in sample_locs[0]:
            for sample_loc_1 in sample_locs[1]:
                patch = ImagePatch( p_id, self, (sample_loc_0, sample_loc_1), size, scale)
                patch.label, patch.matched_roi_idx = \
                            self.get_label_for_patch(patch)
                if patch.label != PASCAL_VOC_BACKGROUND_CLASS:
                    self.non_background_patches.append(patch)
                else:
                    self.background_patches.append(patch)
                    
                patches.append(patch)
                p_id += 1
                
        debug("Compute {} patches".format(p_id-p_id_base))
              
        return p_id
    
# Sample the Pascal VOC dataset 
class MultiscaleSampler(PASCALVOCTrain):
    multi_scales = [1., 1.3, 1.6, 2., 2.3, 2.6, 3.0, 3.3, 3.6, 4., 4.3, 4.6, 5.]
    
    def __init__(self, image_set, year, samples_per_img, sample_height, max_imgs, shuffle, path='.', samples_per_batch=None):
        assert( samples_per_img % samples_per_batch == 0)
        super( MultiscaleSampler, self).__init__(image_set, year, path=path, img_per_batch=samples_per_batch)
        self.num_imgs = min( max_imgs, self.num_images)
        self.samples_per_img = samples_per_img
        self.samples_per_batch = samples_per_batch
        self.sample_height = sample_height
        self.batches_per_img = self.samples_per_img / self.samples_per_batch
        self.shape = (3, sample_height, sample_height)
        self.shuffle = shuffle
        
        assert( self.be.bsz == self.samples_per_batch)
        
        self.dev_X = self.be.iobuf( self.shape, dtype=np.float32)
        self.dev_X_chw = self.dev_X.reshape(3, sample_height, sample_height, self.samples_per_batch)
        self.dev_y_labels_flat = self.be.zeros((1, self.samples_per_batch), dtype=np.int32)
        self.dev_y = self.be.zeros((self.num_classes, self.samples_per_batch), dtype=np.int32)

        print "{} Datatset: # images:{}".format( image_set, self.num_imgs)
        
    def set_mode(self, mode):
        assert mode in ('train', 'test')
        self.mode = mode
        
        self.ndata = self.samples_per_img * self.num_imgs
        self.nbatches = self.batches_per_img * self.num_imgs
            
        return

    def resample_patches(self, img):
        # If we are training then bias the sampling to have as many 
        # non-background patches as possible
        total_num_patches = len(img.non_background_patches) + \
                            len(img.background_patches)
        assert total_num_patches >= self.samples_per_img, "Incorrect patch generation"
        if self.image_set[:5] == 'train':
            if len(img.non_background_patches) > self.samples_per_img:
                all_patches = img.non_background_patches[:self.samples_per_img]
            else:
                if total_num_patches > self.samples_per_img:
                    num_bg_patches_to_keep = len(img.background_patches) - \
                                                (total_num_patches - self.samples_per_img)
                    bg_patches_to_keep = self.be.rng.permutation(len(img.background_patches))
                    bg_patches_to_keep = bg_patches_to_keep[:num_bg_patches_to_keep]
                    img.background_patches = [img.background_patches[i] for i in bg_patches_to_keep]
                all_patches = img.non_background_patches + img.background_patches
        else:
            patches = img.non_background_patches + img.background_patches
            if total_num_patches > self.samples_per_img:
                patches_to_keep = self.be.rng.permutation(total_num_patches)
                patches_to_keep = patches_to_keep[:self.samples_per_img]
                all_patches = [patches[i] for i in patches_to_keep]
            else:
                all_patches = patches
            
        return all_patches
            
    def fill_samples_and_labels(self, img_idx, samples_np, labels_np):
        img_db = self.roi_db[ img_idx] 
        # load and process the image using PIL
        img_num_rois = img_db['num_gt']
        img = PascalImage(img_db['img_file'], 
                          img_db['bb'][:img_num_rois, :],
                          np.squeeze(img_db['gt_classes'][:img_num_rois, :], axis=1))
            
        debug("\n============== Processing: Image {} ({}), Shape:{}, #ROIs: {}={} =================".format(img_idx, img.file_name,
                                     img.shape, img_num_rois, img.labels))
        patches = {}
        p_id_base = 0
        for scale_idx, scale in enumerate(self.multi_scales):
            p_id_base = img.compute_patches_at_scale( scale_idx, scale, p_id_base)
            if p_id_base >= self.samples_per_img and scale_idx > 9:
                break

        debug("Sampled. # Non-BG patches:{} # BG patches:{}".format( 
            len(img.non_background_patches), len(img.background_patches)))
        
        # over-represent the non-background patches during training
        all_patches = self.resample_patches(img)

        assert len(all_patches) == self.samples_per_img
        shuf_idx = self.be.rng.permutation(self.samples_per_img)
        for i in range(self.samples_per_img):
            p_idx = shuf_idx[i]
            p = all_patches[p_idx]
            p_img = img.pil.crop([int(b) for b in p.bbox])
            p_img = p_img.resize( (self.sample_height, self.sample_height), Image.LINEAR)
            if DEBUG and False:
                debug_file_path = '{}_{}.jpg'.format( img.debug_file_path_prefix, p_idx)
                p_img.save(debug_file_path)
                
            # load it to numpy and flip the channel RGB to BGR
            p_img_np = np.array(p_img)[:, :, ::-1]
            # Mean subtract and scale an image
            p_img_np = p_img_np.astype(np.float32, copy=False)
            p_img_np -= FRCN_PIXEL_MEANS
            samples_np[:, :, :, i] = p_img_np.transpose(FRCN_IMG_DIM_SWAP)
                
            labels_np[i] = p.label
            
        return

    def __iter__(self):
        # permute the dataset for the epoch
        if self.shuffle is False:
            self.shuf_idx = [i for i in range(self.num_imgs)]
        else:
            debug("Shuffling images")
            self.shuf_idx = [i for i in self.be.rng.permutation(self.num_imgs)]
            
        samples_np = np.zeros((3, self.sample_height, self.sample_height, self.samples_per_img), 
                               dtype=np.float32)
        labels_np = np.zeros(self.samples_per_img, dtype=np.int32)
            
        for self.batch_index in xrange(self.nbatches):
            start = (self.batch_index % self.batches_per_img) * self.samples_per_batch
            end = start + self.samples_per_batch
            
            if start == 0:
                img_idx = self.batch_index / self.batches_per_img 
                img_idx = self.shuf_idx[img_idx]
                self.fill_samples_and_labels( img_idx, samples_np, labels_np)

            self.dev_X_chw.set( np.ascontiguousarray( samples_np[:, :, :, start:end]))
            self.dev_y_labels_flat[:] = labels_np[start:end].reshape(1, -1)
            self.dev_y[:] = self.be.onehot( self.dev_y_labels_flat, axis=0)
            
            yield self.dev_X, self.dev_y
        return

class ImageScores(Metric):
    """
    Compute the per-class score given the patches of the image
    """
    def __init__(self, be_name, num_classes, num_imgs, samples_per_batch, batches_per_img, exponent):
        self.be_name = be_name
        self.num_classes = num_classes
        self.num_imgs = num_imgs
        self.samples_per_batch = samples_per_batch
        self.batches_per_img = batches_per_img
        self.exponent = exponent
        
        self.scores_batch = self.be.zeros( (self.num_classes, self.samples_per_batch), dtype=np.float32)
        self.scores_imgs = self.be.zeros( (self.num_classes, self.num_imgs), dtype=np.float32)
        self.metric_names = ['Image Object Scores']
        
        self.batch_idx = 0
        self.image_idx = 0
        
        return
    
    def compile_stats_for_image(self):
        scores_img = self.be.sum(self.scores_batch, axis = 1)
        self.scores_imgs[:, self.image_idx] = scores_img #self.be.multiply( scores_img, 
                                                #1.0/float(self.samples_per_batch*self.batches_per_img))
        self.scores_batch[:] = 0
        
        #print "Scores: {}".format(self.scores_imgs[:, self.image_idx].asnumpyarray())
        
        return

    def __call__(self, y, t, calcrange=slice(0, None)):
        exp = self.be.power(y, self.exponent)
        self.scores_batch[:] = self.be.add(exp, self.scores_batch)
        
        # if its the last batch of the curr image
        if (self.batch_idx % self.batches_per_img) == (self.batches_per_img-1):
            self.compile_stats_for_image()
            
        # This is purely for information:
        # Compute the misclassification rate to print on the console
        self.image_idx = self.batch_idx // self.batches_per_img
        y_max = self.be.zeros((1, self.samples_per_batch), dtype=np.float32)
        y_max[:] = self.be.argmax(y, axis=0)
        t_max = self.be.zeros((1, self.samples_per_batch), dtype=np.float32)
        t_max[:] = self.be.argmax(t, axis=0)
        diff = self.be.iobuf(1)
        diff[:] = self.be.not_equal( y_max, t_max)
        mean = diff.get()[:, calcrange].mean()
        
        self.batch_idx += 1
     
        print "Img:{} Batchi:{} Misclassification: {}".format( self.image_idx, self.batch_idx, mean)
        return mean
    
    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """

        if use_07_metric:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
            
    def evaluate_class_predictions(self, roi_db, shuf_idx, classname, class_idx, cls_threshold):
        cls_gt = np.zeros( self.num_imgs, dtype=np.float32) # ground truth
        cls_scores = self.scores_imgs[class_idx, :].get().ravel()
        
        
        # sort by confidence
        cls_imgs_sorted = np.argsort(-cls_scores)
        cls_scores_sorted = -np.sort(-cls_scores)
        img_ids = [shuf_idx[i] for i in cls_imgs_sorted]
        img_names = [None] * self.num_imgs
        
        tp = np.zeros( self.num_imgs)
        fp = np.zeros( self.num_imgs)

        num_gt = 0
        for img_i, img_idx in enumerate(img_ids):
            img_db = roi_db[img_idx]
            cls_gt_img = np.squeeze(img_db['gt_classes'][:img_db['num_gt'],:], axis=1) # ground truth of the img
            cls_gt[img_i] = float(class_idx in cls_gt_img)
            img_names[img_i] = os.path.basename(img_db['img_file'])
            cls_score = cls_scores_sorted[img_i]
            if cls_score >= cls_threshold:
                if cls_gt[img_i] > 0:
                    tp[img_i] = 1
                else:
                    fp[img_i] = 1
    
        print "Images: {}".format(img_ids)
        print "Image Files: {}".format(img_names)
        print "Scores: {}".format(cls_scores_sorted)
        print "GroundTruth: {}".format(cls_gt)
        print "False Positives: {}".format(fp)
        print "True Positives: {}".format(tp)

        #tp = tp[:self.num_imgs//2]
        #fp = fp[:self.num_imgs//2]
        
        # compute precision recall
        num_gt = np.sum(cls_gt)
        num_preds = np.sum(tp + fp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(num_gt) if num_gt > 0 else np.ones(tp.shape)
        prec = tp / (tp + fp) if num_preds > 0 else np.zeros(tp.shape)
        ap = self.voc_ap(rec, prec)
    
        return prec, rec, ap
    
    def evaluation(self, roi_db, shuf_idx, threshold):
        
        ap = np.zeros(PASCAL_VOC_NUM_CLASSES, dtype=np.float32)
        for cls_i, cls in enumerate(PASCAL_VOC_CLASSES):
            if cls_i == PASCAL_VOC_BACKGROUND_CLASS:
                continue
            
            print "======= Class:{} ========".format(cls)
            precision, recall, avg_precision = self.evaluate_class_predictions(roi_db, 
                                         shuf_idx, cls, cls_i, threshold)
            
            print "Precision: mean:{} vals:{} ".format( np.mean(precision), precision)
            print "Recall: mean:{} vals:{}".format( np.mean(recall), recall)
            print "Avg. Precision: {}".format( avg_precision)
            ap[cls_i] = avg_precision

        print "Mean Avg. Precison: {} Values:{}".format(np.mean(ap[1:]), ap)
            
        return

if __name__ == "__main__":
    main()   
        
