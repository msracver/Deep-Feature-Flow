# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Xizhou Zhu, Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/dff_rfcn/cfgs/dff_rfcn_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_batch_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes, draw_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deep Feature Flow demo')
    args = parser.parse_args()
    return args

args = parse_args()

def main():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_flownet_rfcn'
    model = '/../model/rfcn_dff_flownet_vid'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_batch_test_symbol(config)

    # set up class names
    num_classes = 31
    classes = ['airplane', 'antelope', 'bear', 'bicycle',
               'bird', 'bus', 'car', 'cattle',
               'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion',
               'lizard', 'monkey', 'motorcycle', 'rabbit',
               'red_panda', 'sheep', 'snake', 'squirrel',
               'tiger', 'train', 'turtle', 'watercraft',
               'whale', 'zebra']

    # load demo data
    image_names = glob.glob(cur_path + '/../demo/ILSVRC2015_val_00007010/*.JPEG')
    output_dir = cur_path + '/../demo/rfcn_dff_batch/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    key_frame_interval = 10

    #

    data = []
    key_im_tensor = None
    cur_im_tensor = []
    im_info_tensor = []
    image_names_list = []
    image_names_batch = []
    for idx, im_name in enumerate(image_names):
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        if idx % key_frame_interval == 0:
            key_im_tensor = im_tensor
        else:
            cur_im_tensor.append(im_tensor)
        im_info_tensor.append(im_info)
        image_names_batch.append(im_name)
        if (idx+1) % key_frame_interval == 0 or idx == len(image_names) - 1:
            data.append({'data_other': np.concatenate(cur_im_tensor), 'im_info': np.concatenate(im_info_tensor), 'data_key': key_im_tensor})
            key_im_tensor = None
            cur_im_tensor = []
            im_info_tensor = []
            image_names_list.append(image_names_batch)
            image_names_batch = []

    # get predictor
    data_names = ['data_other', 'im_info', 'data_key']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data_other', (key_frame_interval-1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),
                       ('data_key', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + model, 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    # warm up
    for j in xrange(1):
        data_batch = mx.io.DataBatch(data=[data[j]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[j])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[:, 2] for i in xrange(len(data_batch.data))]
        scores_all, boxes_all, data_dict = im_batch_detect(predictor, data_batch, data_names, scales, config)

    print "warmup done"
    # test
    time = 0
    count = 0
    for idx, im_names in enumerate(image_names_list):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[:, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores_all, boxes_all, data_dict = im_batch_detect(predictor, data_batch, data_names, scales, config)
        time += toc()
        count += len(scores_all)
        print 'testing {} {:.4f}s x {:d}'.format(im_names[0], time/count, len(scores_all))

        for batch_idx in xrange(len(scores_all)):
            boxes = boxes_all[batch_idx].astype('f')
            scores = scores_all[batch_idx].astype('f')
            dets_nms = []
            for j in range(1, scores.shape[1]):
                cls_scores = scores[:, j, np.newaxis]
                cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                cls_dets = cls_dets[keep, :]
                cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
                dets_nms.append(cls_dets)
            # visualize
            im = cv2.imread(im_names[batch_idx])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # show_boxes(im, dets_nms, classes, 1)
            out_im = draw_boxes(im, dets_nms, classes, 1)
            _, filename = os.path.split(im_names[batch_idx])
            cv2.imwrite(output_dir + filename,out_im)


    print 'done'

if __name__ == '__main__':
    main()
