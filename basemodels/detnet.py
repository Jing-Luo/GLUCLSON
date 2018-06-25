# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""DetNet, implemented in Gluon."""

__all__ = ['DetNet', 'DilatedBottleneck', 'detnet59', 'get_detnet']

__modify__ = 'BJJ'
__modified_date__ = '18/05/18'

import os

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


# Blocks
class DilatedBottleneck(HybridBlock):
    def __init__(self, channels, stride, dilation, downsample, **kwargs):
        super(DilatedBottleneck, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels//4, kernel_size=3, strides=1, dilation=dilation, padding=dilation))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x

# Nets
class DetNet(HybridBlock):
    r"""DetNet model from
    `"DetNet: A Backbone Network for Object Detection"
    <http://arxiv.org/abs/1804.06215>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block.
    dilations : list of int
        Dilation value in each block.
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, layers, dilations, channels, classes=1000, **kwargs):
        super(DetNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(num_layer, channels[i+1], stride, dilations[i], i+1))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, layers, channels, stride, dilation, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(DilatedBottleneck(channels, stride, dilation, True, prefix=''))
            for _ in range(layers-1):
                layer.add(DilatedBottleneck(channels, 1, dilation, False, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x

# Specification
detnet_spec = {59: ([3, 4, 6, 3, 3], [1, 1, 1, 2, 2], [64, 256, 512, 1024, 1024, 1024])}

def get_detnet(num_layers, pretrained=False, ctx=cpu(),
               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""DetNet model from
    `"DetNet: A Backbone Network for Object Detection"
    <http://arxiv.org/abs/1804.06215>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Options are 59.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    layers, dilations, channels = detnet_spec[num_layers]
    net = DetNet(layers, dilations, channels, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('resnet%d_v%d'%(num_layers, version),
                                       root=root), ctx=ctx)
    return net

def detnet59(**kwargs):
    r"""DetNet-59 model from
    `"DetNet: A Backbone Network for Object Detection"
    <http://arxiv.org/abs/1804.06215>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_detnet(59, **kwargs)
