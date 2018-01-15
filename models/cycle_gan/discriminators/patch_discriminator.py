# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow
import tensorflow as tf
import numpy as np
from ...utils.layers import *
from .registry import register


@register("patch_discriminator")
def patch_discriminator(x, d_params, m_params):

  with tf.variable_scope(d_params.network_name):
    filter_size = 4
    num_discrim_filters = 64
    patch_input = tf.random_crop(x, [1, 70, 70, m_params.input_shape[2]])
    return build_n_layer_conv_stack(general_conv2d, patch_input, filter_size,
                                    num_discrim_filters)
