import tensorflow as tf

from .registry import register
from .defaults import default


@register("gan_default")
def gan_default():
  gan_default = default()
  F = tf.contrib.training.HParams(name="cyclegan_generator", network_name="F")
  G = tf.contrib.training.HParams(name="cyclegan_generator", network_name="G")
  gan_default.add_hparam("F", F)
  gan_default.add_hparam("G", G)
  discriminator_X = tf.contrib.training.HParams(
      name="patch_discriminator", network_name="discriminator_X")
  discriminator_Y = tf.contrib.training.HParams(
      name="patch_discriminator", network_name="discriminator_Y")
  gan_default.add_hparam("discriminator_X", discriminator_X)
  gan_default.add_hparam("discriminator_Y", discriminator_Y)
  gan_default.add_hparam("cycle_loss", 10)
  gan_default.add_hparam("past_count", 50)
  gan_default.add_hparam("use_wasserstein", True)
  gan_default.add_hparam("wasserstein_loss", 10)
  gan_default.add_hparam("norm_type", None)
  gan_default.add_hparam("discrim_loss", None)
  gan_default.add_hparam("d_step", 1)
  gan_default.add_hparam("original_l2", False)
  return gan_default


@register("hz_cg")
def hz_cg():
  hz_cg = gan_default()
  hz_cg.add_hparam("input_shape", [256, 256, 3])
  hz_cg.add_hparam("output_shape", [256, 256, 3])
  hz_cg.learning_rate = 2e-4
  hz_cg.lr_scheme = "delay_lin"
  hz_cg.delay = 100000
  hz_cg.batch_size = 1
  return hz_cg


@register("ns_cg")
def ns_cg():
  ns_cg = gan_default()
  ns_cg.add_hparam("input_shape", [257, 251, 1])
  ns_cg.add_hparam("output_shape", [257, 251, 1])
  ns_cg.learning_rate = 2e-4
  ns_cg.lr_scheme = "delay_lin"
  ns_cg.delay = 100000
  ns_cg.batch_size = 1
  ns_cg.type = "audio_timbre"
  return ns_cg