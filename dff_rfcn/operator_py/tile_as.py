# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

import mxnet as mx
import numpy as np
from distutils.util import strtobool

class TileAsOperator(mx.operator.CustomOp):
    def __init__(self):
        super(TileAsOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        data_content = in_data[0]
        data_tiled = mx.ndarray.tile(data_content, reps=(in_data[1].shape[0], 1, 1, 1))
        self.assign(out_data[0], req[0], data_tiled)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('tile_as')
class TileAsProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(TileAsProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data_content', 'data_shape']

    def list_outputs(self):
        return ['data_tiled']

    def infer_shape(self, in_shape):
        data_content_shape = in_shape[0]
        data_shape_shape = in_shape[1]

        tiled_data_shape = (data_shape_shape[0], data_content_shape[1], data_content_shape[2], data_content_shape[3])

        return [data_content_shape, data_shape_shape], \
               [tiled_data_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return TileAsOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return out_grad
