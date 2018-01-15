import tensorflow as tf

from .registry import register
from ...utils.layers import *


# modified from https://github.com/hardikbansal/CycleGAN/blob/master/models.py
@register("cyclegan_generator")
def build_generator_resnet_9blocks(x, g_params, m_params):
  with tf.variable_scope(g_params.network_name):
    filtersize = 3
    img_layer = m_params.input_shape[2]
    batch_size = 1
    num_generator_filter = 32
    pad_input = tf.pad(x, [[0, 0], [filtersize, filtersize],
                           [filtersize, filtersize], [0, 0]], "REFLECT")
    o_c1 = general_conv2d(
        x=pad_input,
        num_filters=num_generator_filter,
        filter_size=7,
        stride=1,
        stddev=0.02,
        name="c1")
    if (m_params.type == "audio_timbre"):
      o_c2 = general_conv2d(
          x=o_c1,
          num_filters=num_generator_filter * 2,
          filter_size=filtersize,
          stride=2,
          stddev=0.02,
          padding="VALID",
          name="c2")
    else:
      o_c2 = general_conv2d(
          x=o_c1,
          num_filters=num_generator_filter * 2,
          filter_size=filtersize,
          stride=2,
          stddev=0.02,
          padding="SAME",
          name="c2")
    o_c3 = general_conv2d(
        x=o_c2,
        num_filters=num_generator_filter * 4,
        filter_size=filtersize,
        stride=2,
        stddev=0.02,
        padding="SAME",
        name="c3")

    o_r1 = build_resnet_block(
        general_conv2d, x=o_c3, dim=num_generator_filter * 4, name="r1")
    o_r2 = build_resnet_block(
        general_conv2d, x=o_r1, dim=num_generator_filter * 4, name="r2")
    o_r3 = build_resnet_block(
        general_conv2d, x=o_r2, dim=num_generator_filter * 4, name="r3")
    o_r4 = build_resnet_block(
        general_conv2d, x=o_r3, dim=num_generator_filter * 4, name="r4")
    o_r5 = build_resnet_block(
        general_conv2d, x=o_r4, dim=num_generator_filter * 4, name="r5")
    o_r6 = build_resnet_block(
        general_conv2d, x=o_r5, dim=num_generator_filter * 4, name="r6")
    o_r7 = build_resnet_block(
        general_conv2d, x=o_r6, dim=num_generator_filter * 4, name="r7")
    o_r8 = build_resnet_block(
        general_conv2d, x=o_r7, dim=num_generator_filter * 4, name="r8")
    o_r9 = build_resnet_block(
        general_conv2d, x=o_r8, dim=num_generator_filter * 4, name="r9")
    o_c4 = general_deconv2d(
        x=o_r9,
        outshape=[batch_size, 128, 128, num_generator_filter * 2],
        num_filters=num_generator_filter * 2,
        filter_size=filtersize,
        stride=2,
        stddev=0.02,
        padding="SAME",
        name="c4")
    if (m_params.type == "audio_timbre"):
      o_c5 = general_deconv2d(
          x=o_c4,
          outshape=[batch_size, 256, 256, num_generator_filter],
          num_filters=num_generator_filter,
          filter_size=filtersize,
          stride=2,
          stddev=0.02,
          padding="VALID",
          name="c5")
    else:
      o_c5 = general_deconv2d(
          x=o_c4,
          outshape=[batch_size, 256, 256, num_generator_filter],
          num_filters=num_generator_filter,
          filter_size=filtersize,
          stride=2,
          stddev=0.02,
          padding="SAME",
          name="c5")
    o_c6 = general_conv2d(
        x=o_c5,
        num_filters=img_layer,
        filter_size=7,
        stride=1,
        stddev=0.02,
        padding="SAME",
        name="c6",
        do_relu=False)

    # Adding the tanh layer
    out_gen = tf.nn.tanh(o_c6)
    if (m_params.type == "audio_timbre"):
      out_gen = tf.slice(out_gen, [0, 0, 0, 0], [-1, -1, 251, -1])

    return out_gen