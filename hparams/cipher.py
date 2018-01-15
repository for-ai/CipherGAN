import tensorflow as tf

from .registry import register
from .gan import gan_default


@register("cipher_default")
def cipher_default():
  cipher_default = gan_default()
  return cipher_default


@register("simple_cipher")
def simple_cipher():
  c = cipher_default()
  c.cycle_loss = 100
  c.learning_rate = 2e-4
  c.lr_scheme = "delay_lin"
  c.delay = 100000
  c.type = "text"
  c.add_hparam("lp_distance", "l1")
  c.F.name = "sequence_generator"
  c.F.add_hparam("filter_count", 32)
  c.F.add_hparam("filter_size", 1)
  c.F.add_hparam("add_timing", False)
  c.G = tf.contrib.training.HParams.from_proto(c.F.to_proto())
  c.G.network_name = "G"
  c.discriminator_X.name = "sequence_discriminator"
  c.discriminator_X.add_hparam("filter_count", 32)
  c.discriminator_X.add_hparam("filter_size", 15)
  c.discriminator_X.add_hparam("add_timing", False)
  c.discriminator_X.add_hparam("dropout", 0.0)
  c.discriminator_Y = tf.contrib.training.HParams.from_proto(
      c.discriminator_X.to_proto())
  c.discriminator_Y.network_name = "discriminator_Y"

  c.add_hparam("sample_length", 100)
  c.add_hparam("vocab_size", 26)
  c.add_hparam("vocab", [str(i) for i in range(c.vocab_size)])
  c.add_hparam("hidden_size", 100)
  c.add_hparam("input_shape", [c.sample_length, c.vocab_size])

  c.use_wasserstein = True
  c.discrim_loss = None  # 'full_wgan' or 'log', otherwise partial
  c.wasserstein_loss = 10
  c.norm_type = "layer"
  c.add_hparam("true_lipschitz", False)

  c.add_hparam("use_embeddings", True)
  c.add_hparam("timing_type", "transformer")

  c.add_hparam("optimizer", "adam")
  c.add_hparam("beta1", 0.9)
  c.add_hparam("beta2", 0.999)
  return c
