import tensorflow as tf

from .registry import register


@register("synthetic")
def input_fn(data_sources, params, training):

  def _input_fn():
    x = tf.random_uniform(
        [params.batch_size] + params.input_shape, minval=0, maxval=1)
    output = tf.random_uniform(
        [params.batch_size] + params.output_shape, minval=0, maxval=1)

    return {"inputs": x}, output

  return _input_fn