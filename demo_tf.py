#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""


import _init_paths_tf
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network
from fast_rcnn.nms_wrapper import nms
import pprint
import tensorflow as tf
import time, os, sys, fire, cv2
from utils.timer import Timer
from fast_rcnn.test import im_detect

import numpy as np
import cPickle


CLASSES = ('__background__', 'tower', 'insulator', 'hammer', 'nest', 'text')

COLOR = {'tower': (0, 255, 0), 'insulator':(0, 0, 255), 'nest': (255, 0, 255)}

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)

def init( model= os.path.join(cfg.ROOT_DIR, "output/faster_rcnn_end2end/result/insulator_2016_trainval_exp1/VGGnet_fast_rcnn_iter_70000.ckpt"),
        imdb_name='insulator_2016_test', 
        net_name='VGGnet_test' ):

    global sess
    imdb = get_imdb(imdb_name)
    net = get_network(net_name, imdb.num_classes)
    
    if not os.path.exists(model+'.meta'):
        print "{} not exist".format(model)
        return
    
    cfg.USE_GPU_NMS = True
    cfg.GPU_ID = 0
    
    # load weights
    saver = tf.train.Saver()
    saver.restore(sess, model)
    print ('Loading model weights from {:s}').format(model)


    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _, _= im_detect(sess, net, im)
    return net

def detect(net, imgPath):
    global sess
    im = cv2.imread(imgPath)
    timer = Timer()
    timer.tic()

    scores, boxes, _ = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals, {}').format(timer.total_time, boxes.shape[0], imgPath)

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background

        if cls == 'text':
            continue

        if cls == 'hammer':
            continue

        if cls == 'tower':
            continue

        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
    
        thresh = 0.5
        inds = np.where(dets[:, -1] >= thresh)[0]
        for i in inds:
            bbox = dets[i, :4]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR[cls], 3)
    return im
