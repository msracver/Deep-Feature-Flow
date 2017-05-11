# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

import mxnet as mx
import numpy as np
from distutils.util import strtobool

class RPNInvNormalizeOperator(mx.operator.CustomOp):
    def __init__(self, num_anchors, bbox_mean, bbox_std):
        super(RPNInvNormalizeOperator, self).__init__()
        self._num_anchors = num_anchors
        self._bbox_mean = mx.ndarray.Reshape(mx.nd.array(bbox_mean), shape=(1,4,1,1))
        self._bbox_std = mx.ndarray.Reshape(mx.nd.array(bbox_std), shape=(1,4,1,1))

    def forward(self, is_train, req, in_data, out_data, aux):
        bbox_pred = in_data[0]
        tile_shape = (bbox_pred.shape[0], self._num_anchors, bbox_pred.shape[2], bbox_pred.shape[3])
        bbox_mean = mx.ndarray.tile(self._bbox_mean.as_in_context(bbox_pred.context), reps=tile_shape)
        bbox_std = mx.ndarray.tile(self._bbox_std.as_in_context(bbox_pred.context), reps=tile_shape)
        bbox_pred = bbox_pred * bbox_std + bbox_mean

        self.assign(out_data[0], req[0], bbox_pred)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register('rpn_inv_normalize')
class RPNInvNormalizeProp(mx.operator.CustomOpProp):
    def __init__(self, num_anchors, bbox_mean='(0.0, 0.0, 0.0, 0.0)', bbox_std='0.1, 0.1, 0.2, 0.2'):
        super(RPNInvNormalizeProp, self).__init__(need_top_grad=False)
        self._num_anchors = int(num_anchors)
        self._bbox_mean = np.fromstring(bbox_mean[1:-1], dtype=float, sep=',')
        self._bbox_std  = np.fromstring(bbox_std[1:-1], dtype=float, sep=',')

    def list_arguments(self):
        return ['bbox_pred']

    def list_outputs(self):
        return ['out_bbox_pred']

    def infer_shape(self, in_shape):

        return [in_shape[0]], \
               [in_shape[0]]

    def create_operator(self, ctx, shapes, dtypes):
        return RPNInvNormalizeOperator(self._num_anchors, self._bbox_mean, self._bbox_std)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
