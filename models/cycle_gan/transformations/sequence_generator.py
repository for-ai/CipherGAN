import tensorflow as tf

from .registry import register
from ...utils.layers import *
from ...utils.model_utils import get_embedding_var


@register("simple_generator")
def sequence_generator(x, g_params, m_params):
  with tf.variable_scope(g_params.network_name):
    x = general_conv1d(
        x,
        num_filters=g_params.filter_count,
        filter_size=g_params.filter_size,
        stride=1,
        stddev=0.02,
        do_norm='layer',
        do_relu=True,
        relufactor=0,
        name="c1")
    x = tf.nn.relu(x + general_conv1d(
        x,
        num_filters=g_params.filter_count,
        filter_size=g_params.filter_size,
        stride=1,
        stddev=0.02,
        do_norm='layer',
        do_relu=False,
        relufactor=0,
        name="c2"))
    x = general_conv1d(
        x,
        num_filters=m_params.vocab_size,
        filter_size=g_params.filter_size,
        stride=1,
        stddev=0.02,
        do_norm=None,
        do_relu=False,
        relufactor=0,
        name="c3")

    # Softmax layer defines a probability distribution over output vocabulary
    output_dist = tf.nn.softmax(x, dim=-1)

    return output_dist


# modified from https://github.com/hardikbansal/CycleGAN/blob/master/models.py
@register("sequence_generator")
def sequence_generator(x, g_params, m_params):
  with tf.variable_scope(g_params.network_name):
    if g_params.add_timing:
      x = timing(x, m_params)

    pad_input = tf.pad(
        x, [[0, 0], [g_params.filter_size // 2, g_params.filter_size // 2],
            [0, 0]], "CONSTANT")
    c = general_conv1d(
        pad_input,
        num_filters=g_params.filter_count,
        filter_size=g_params.filter_size,
        stride=1,
        stddev=0.02,
        name="c1")
    c = general_conv1d(
        c,
        num_filters=g_params.filter_count * 2,
        filter_size=g_params.filter_size,
        stride=1,
        stddev=0.02,
        padding="SAME",
        name="c2")
    l = general_conv1d(
        c,
        num_filters=g_params.filter_count * 4,
        filter_size=g_params.filter_size,
        stride=1,
        stddev=0.02,
        padding="SAME",
        name="c3")

    for i in range(5):
      l = build_resnet_block(
          general_conv1d,
          l,
          dim=g_params.filter_count * 4,
          filter_size=g_params.filter_size,
          pad=True,
          name="r%d" % i)

    c = general_conv1d(
        l,
        num_filters=m_params.vocab_size,
        filter_size=1,
        stride=1,
        stddev=0.02,
        padding="SAME",
        name="out")

    # Softmax layer defines a probability distribution over output vocabulary
    output_dist = tf.nn.softmax(c, dim=-1)

    return output_dist
