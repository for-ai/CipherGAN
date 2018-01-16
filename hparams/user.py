import tensorflow as tf

from .registry import register
from .cipher import simple_cipher


# settings for Vigenere cipher vocab size 200
@register("vigenere_brown_vocab_200")
def vigenere():
  c = simple_cipher()
  c.use_embeddings = True
  c.timing_type = "concat"
  c.batch_size = 64
  c.cycle_loss = 1
  c.add_hparam("vocab_file", "vigenere345_brown200_vocab.txt")
  c.lp_distance = "l1"

  c.hidden_size = 256

  c.use_wasserstein = True
  c.wasserstein_loss = 10
  c.original_l2 = True

  c.discrim_loss = None

  c.lr_scheme = "warmup_constant"
  c.warmup_steps = 2500
  c.learning_rate = 2e-4
  c.beta1 = 0
  c.beta2 = 0.9

  c.F.add_timing = True
  c.F.filter_size = 1
  c.G = tf.contrib.training.HParams.from_proto(c.F.to_proto())
  c.G.network_name = "G"

  c.discriminator_X.add_timing = True
  c.discriminator_X.dropout = 0.5
  c.discriminator_X.filter_count = 32
  c.discriminator_Y = tf.contrib.training.HParams.from_proto(
    c.discriminator_X.to_proto())
  c.discriminator_Y.network_name = "discriminator_Y"

  return c
