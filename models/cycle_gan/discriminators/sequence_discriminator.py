import tensorflow as tf

from ...utils.layers import *
from ...utils.model_utils import get_embedding_var
from .registry import register


@register("sequence_discriminator")
def sequence_discriminator(x, d_params, m_params):
  with tf.variable_scope(d_params.network_name):
    if d_params.add_timing:
      x = timing(x, m_params)

    if d_params.dropout != 0:
      x = tf.nn.dropout(x, 1 - d_params.dropout)

    x = build_n_layer_conv_stack(
        general_conv1d,
        x,
        d_params.filter_size,
        d_params.filter_count,
        n=5,
        do_norm=m_params.norm_type)

    if m_params.discrim_loss == "log":
      x = tf.nn.softplus(x)

    return x
