import tensorflow as tf

from .registry import register


@register("default")
def default():
  return tf.contrib.training.HParams(
      type="image",
      batch_size=64,
      learning_rate=0.01,
      lr_scheme="exp",
      delay=0,
      staircased=False,
      learning_rate_decay_interval=2000,
      learning_rate_decay_rate=0.1,
      clip_grad_norm=1.0,
      l2_loss=0.0,
      label_smoothing=0.1,
      init_scheme="random",
      warmup_steps=10000)