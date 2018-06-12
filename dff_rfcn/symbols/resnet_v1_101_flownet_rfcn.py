# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.rpn_inv_normalize import *
from operator_py.tile_as import *

class resnet_v1_101_flownet_rfcn(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_v1(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1 , act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu , pad=(1,1), kernel=(3,3), stride=(2,2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1 , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1 , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a , act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b , act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1,scale2a_branch2c] )
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a , act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a , act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b , act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu,scale2b_branch2c] )
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b , act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a , act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b , act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu,scale2c_branch2c] )
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c , act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a , act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b , act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1,scale3a_branch2c] )
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a , act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a , act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b , act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu,scale3b1_branch2c] )
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1 , act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a , act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b , act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu,scale3b2_branch2c] )
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2 , act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a , act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b , act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu,scale3b3_branch2c] )
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3 , act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a , act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b , act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1,scale4a_branch2c] )
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a , act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a , act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b , act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu,scale4b1_branch2c] )
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1 , act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a , act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b , act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu,scale4b2_branch2c] )
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2 , act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a , act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b , act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu,scale4b3_branch2c] )
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3 , act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a , act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b , act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu,scale4b4_branch2c] )
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4 , act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a , act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b , act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu,scale4b5_branch2c] )
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5 , act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a , act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b , act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu,scale4b6_branch2c] )
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6 , act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a , act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b , act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu,scale4b7_branch2c] )
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7 , act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a , act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b , act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu,scale4b8_branch2c] )
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8 , act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a , act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b , act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu,scale4b9_branch2c] )
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9 , act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a , act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b , act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu,scale4b10_branch2c] )
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10 , act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a , act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b , act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu,scale4b11_branch2c] )
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11 , act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a , act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b , act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu,scale4b12_branch2c] )
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12 , act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a , act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b , act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu,scale4b13_branch2c] )
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13 , act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a , act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b , act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu,scale4b14_branch2c] )
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14 , act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a , act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b , act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu,scale4b15_branch2c] )
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15 , act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a , act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b , act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu,scale4b16_branch2c] )
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16 , act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a , act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b , act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu,scale4b17_branch2c] )
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17 , act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a , act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b , act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu,scale4b18_branch2c] )
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18 , act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a , act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b , act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu,scale4b19_branch2c] )
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19 , act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a , act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b , act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu,scale4b20_branch2c] )
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20 , act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a , act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b , act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu,scale4b21_branch2c] )
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21 , act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a , act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b , act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu,scale4b22_branch2c] )
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22 , act_type='relu')
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b22_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b22_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a , act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu , num_filter=512, pad=(2,2), dilate=(2,2), kernel=(3,3), stride=(1,1), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b , act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1,scale5a_branch2c] )
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a , act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a , act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu , num_filter=512, pad=(2,2), dilate=(2,2), kernel=(3,3), stride=(1,1), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b , act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu,scale5b_branch2c] )
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b , act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a , act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu , num_filter=512, pad=(2,2), dilate=(2,2), kernel=(3,3), stride=(1,1), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b , act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu,scale5c_branch2c] )
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c , act_type='relu')

        feat_conv_3x3 = mx.sym.Convolution(
            data=res5c_relu, kernel=(3, 3), pad=(6, 6), dilate=(6, 6), num_filter=1024, name="feat_conv_3x3")
        feat_conv_3x3_relu = mx.sym.Activation(data=feat_conv_3x3, act_type="relu", name="feat_conv_3x3_relu")
        return feat_conv_3x3_relu

    def get_flownet(self, img_cur, img_ref):
        data = mx.symbol.Concat(img_cur / 255.0, img_ref / 255.0, dim=1)
        resize_data = mx.symbol.Pooling(name='resize_data', data=data , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        flow_conv1 = mx.symbol.Convolution(name='flow_conv1', data=resize_data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=False)
        ReLU1 = mx.symbol.LeakyReLU(name='ReLU1', data=flow_conv1 , act_type='leaky', slope=0.1)
        conv2 = mx.symbol.Convolution(name='conv2', data=ReLU1 , num_filter=128, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
        ReLU2 = mx.symbol.LeakyReLU(name='ReLU2', data=conv2 , act_type='leaky', slope=0.1)
        conv3 = mx.symbol.Convolution(name='conv3', data=ReLU2 , num_filter=256, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
        ReLU3 = mx.symbol.LeakyReLU(name='ReLU3', data=conv3 , act_type='leaky', slope=0.1)
        conv3_1 = mx.symbol.Convolution(name='conv3_1', data=ReLU3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU4 = mx.symbol.LeakyReLU(name='ReLU4', data=conv3_1 , act_type='leaky', slope=0.1)
        conv4 = mx.symbol.Convolution(name='conv4', data=ReLU4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU5 = mx.symbol.LeakyReLU(name='ReLU5', data=conv4 , act_type='leaky', slope=0.1)
        conv4_1 = mx.symbol.Convolution(name='conv4_1', data=ReLU5 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU6 = mx.symbol.LeakyReLU(name='ReLU6', data=conv4_1 , act_type='leaky', slope=0.1)
        conv5 = mx.symbol.Convolution(name='conv5', data=ReLU6 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU7 = mx.symbol.LeakyReLU(name='ReLU7', data=conv5 , act_type='leaky', slope=0.1)
        conv5_1 = mx.symbol.Convolution(name='conv5_1', data=ReLU7 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU8 = mx.symbol.LeakyReLU(name='ReLU8', data=conv5_1 , act_type='leaky', slope=0.1)
        conv6 = mx.symbol.Convolution(name='conv6', data=ReLU8 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU9 = mx.symbol.LeakyReLU(name='ReLU9', data=conv6 , act_type='leaky', slope=0.1)
        conv6_1 = mx.symbol.Convolution(name='conv6_1', data=ReLU9 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU10 = mx.symbol.LeakyReLU(name='ReLU10', data=conv6_1 , act_type='leaky', slope=0.1)
        Convolution1 = mx.symbol.Convolution(name='Convolution1', data=ReLU10 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv5 = mx.symbol.Deconvolution(name='deconv5', data=ReLU10 , num_filter=512, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv5 = mx.symbol.Crop(name='crop_deconv5', *[deconv5,ReLU8] , offset=(1,1))
        ReLU11 = mx.symbol.LeakyReLU(name='ReLU11', data=crop_deconv5 , act_type='leaky', slope=0.1)
        upsample_flow6to5 = mx.symbol.Deconvolution(name='upsample_flow6to5', data=Convolution1 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow6_to_5 = mx.symbol.Crop(name='crop_upsampled_flow6_to_5', *[upsample_flow6to5,ReLU8] , offset=(1,1))
        Concat2 = mx.symbol.Concat(name='Concat2', *[ReLU8,ReLU11,crop_upsampled_flow6_to_5] )
        Convolution2 = mx.symbol.Convolution(name='Convolution2', data=Concat2 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv4 = mx.symbol.Deconvolution(name='deconv4', data=Concat2 , num_filter=256, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv4 = mx.symbol.Crop(name='crop_deconv4', *[deconv4,ReLU6] , offset=(1,1))
        ReLU12 = mx.symbol.LeakyReLU(name='ReLU12', data=crop_deconv4 , act_type='leaky', slope=0.1)
        upsample_flow5to4 = mx.symbol.Deconvolution(name='upsample_flow5to4', data=Convolution2 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow5_to_4 = mx.symbol.Crop(name='crop_upsampled_flow5_to_4', *[upsample_flow5to4,ReLU6] , offset=(1,1))
        Concat3 = mx.symbol.Concat(name='Concat3', *[ReLU6,ReLU12,crop_upsampled_flow5_to_4] )
        Convolution3 = mx.symbol.Convolution(name='Convolution3', data=Concat3 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv3 = mx.symbol.Deconvolution(name='deconv3', data=Concat3 , num_filter=128, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv3 = mx.symbol.Crop(name='crop_deconv3', *[deconv3,ReLU4] , offset=(1,1))
        ReLU13 = mx.symbol.LeakyReLU(name='ReLU13', data=crop_deconv3 , act_type='leaky', slope=0.1)
        upsample_flow4to3 = mx.symbol.Deconvolution(name='upsample_flow4to3', data=Convolution3 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow4_to_3 = mx.symbol.Crop(name='crop_upsampled_flow4_to_3', *[upsample_flow4to3,ReLU4] , offset=(1,1))
        Concat4 = mx.symbol.Concat(name='Concat4', *[ReLU4,ReLU13,crop_upsampled_flow4_to_3] )
        Convolution4 = mx.symbol.Convolution(name='Convolution4', data=Concat4 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv2 = mx.symbol.Deconvolution(name='deconv2', data=Concat4 , num_filter=64, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv2 = mx.symbol.Crop(name='crop_deconv2', *[deconv2,ReLU2] , offset=(1,1))
        ReLU14 = mx.symbol.LeakyReLU(name='ReLU14', data=crop_deconv2 , act_type='leaky', slope=0.1)
        upsample_flow3to2 = mx.symbol.Deconvolution(name='upsample_flow3to2', data=Convolution4 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow3_to_2 = mx.symbol.Crop(name='crop_upsampled_flow3_to_2', *[upsample_flow3to2,ReLU2] , offset=(1,1))
        Concat5 = mx.symbol.Concat(name='Concat5', *[ReLU2,ReLU14,crop_upsampled_flow3_to_2] )
        Concat5 = mx.symbol.Pooling(name='resize_concat5', data=Concat5 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        Convolution5 = mx.symbol.Convolution(name='Convolution5', data=Concat5 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)

        Convolution5_scale_bias = mx.sym.Variable(name='Convolution5_scale_bias', lr_mult=0.0)
        Convolution5_scale = mx.symbol.Convolution(name='Convolution5_scale', data=Concat5 , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1),
                                                   bias=Convolution5_scale_bias, no_bias=False)
        return Convolution5 * 2.5, Convolution5_scale

    def get_train_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data = mx.sym.Variable(name="data")
        data_ref = mx.sym.Variable(name="data_ref")
        eq_flag = mx.sym.Variable(name="eq_flag")
        im_info = mx.sym.Variable(name="im_info")
        gt_boxes = mx.sym.Variable(name="gt_boxes")
        rpn_label = mx.sym.Variable(name='label')
        rpn_bbox_target = mx.sym.Variable(name='bbox_target')
        rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

        # shared convolutional layers
        conv_feat = self.get_resnet_v1(data_ref)
        flow, scale_map = self.get_flownet(data, data_ref)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        warp_conv_feat = mx.sym.BilinearSampler(data=conv_feat, grid=flow_grid, name='warping_feat')
        warp_conv_feat = warp_conv_feat * scale_map
        select_conv_feat = mx.sym.take(mx.sym.Concat(*[warp_conv_feat, conv_feat], dim=0), eq_flag)

        conv_feats = mx.sym.SliceChannel(select_conv_feat, axis=1, num_outputs=2)

        # RPN layers
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        # prepare rpn data
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

        # classification
        rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                               normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
        # bounding box regression
        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)
        else:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

        # ROI proposal
        rpn_cls_act = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
        rpn_cls_act_reshape = mx.sym.Reshape(
            data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

        if cfg.TRAIN.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

         # ROI proposal target
        gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
        rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))


        # classification
        if cfg.TRAIN.ENABLE_OHEM:
            print 'use ohem!'
            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
            bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
            rcnn_label = labels_ohem
        else:
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
            rcnn_label = label

        # reshape output
        rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')

        group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        self.sym = group
        return group

    def get_key_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")
        data_key = mx.sym.Variable(name="data_key")
        feat_key = mx.sym.Variable(name="feat_key")

        # shared convolutional layers
        conv_feat = self.get_resnet_v1(data)
        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([data_key, feat_key, conv_feat, rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def get_cur_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_cur = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")
        data_key = mx.sym.Variable(name="data_key")
        conv_feat = mx.sym.Variable(name="feat_key")

        # shared convolutional layers
        flow, scale_map = self.get_flownet(data_cur, data_key)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        conv_feat = mx.sym.BilinearSampler(data=conv_feat, grid=flow_grid, name='warping_feat')
        conv_feat = conv_feat * scale_map
        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def get_batch_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_key = mx.sym.Variable(name="data_key")
        data_other = mx.sym.Variable(name="data_other")
        im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat_key = self.get_resnet_v1(data_key)

        data_key_tiled = mx.sym.Custom(data_content=data_key, data_shape=data_other, op_type='tile_as')
        conv_feat_key_tiled = mx.sym.Custom(data_content=conv_feat_key, data_shape=data_other, op_type='tile_as')
        flow, scale_map = self.get_flownet(data_other, data_key_tiled)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        conv_feat_other = mx.sym.BilinearSampler(data=conv_feat_key_tiled, grid=flow_grid, name='warping_feat')
        conv_feat_other = conv_feat_other * scale_map

        conv_feat = mx.symbol.Concat(conv_feat_key, conv_feat_other, dim=0)

        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.MultiProposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            NotImplemented

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        arg_params['Convolution5_scale_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['Convolution5_scale_weight'])
        arg_params['Convolution5_scale_bias'] = mx.nd.ones(shape=self.arg_shape_dict['Convolution5_scale_bias'])

        arg_params['feat_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['feat_conv_3x3_weight'])
        arg_params['feat_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['feat_conv_3x3_bias'])

        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])
