"""
Generates text samples and saves them as TFRecords.

Note:
  - separate_domain: It shifts the cipher domain out of the plaintext domain.
    i.e. plaintext - [1,2,3] and encipher using shift or Vigenere it will result in
    something in the same domain ([1,2,3]) so if you set separate_domain to True it
    will add 3 to every cipher text element putting it all in [4,5,6]. To improve
    discriminator's performance.
  - corpus
    - custom: no padding needed.
    - non-custom: index 0 and index 1 are <pad> and <unk>

See tf.flags defined below for all available options.
"""

import nltk
from collections import deque
from generator_utils import *

tf.flags.DEFINE_string("output_dir", "tmp/dataset/cipher",
                       "The output directory to write data to.")
tf.flags.DEFINE_string("train_name", "data-train",
                       "The filename to store training samples under.")
tf.flags.DEFINE_string("test_name", "data-eval",
                       "The filename to store testing samples under.")
tf.flags.DEFINE_string("vocab_filename", "vocab.txt",
                       "The filename to write the vocabulary to.")
tf.flags.DEFINE_string(
    "corpus", "custom",
    "Choice of nltk corpus to use. If 'custom' uses vocab defined in plain_vocab. "
    "A full list of available corpii is available at http://www.nltk.org/nltk_data/"
)
tf.flags.DEFINE_string("cipher", "shift", "Choice of shift or vigenere")
tf.flags.DEFINE_bool("separate_domains", False,
                     "Whether input and output domains should be separated.")
tf.flags.DEFINE_float(
    "percentage_training", .8,
    "What percentage of the corpus should be used for training and the rest for testing"
)
tf.flags.DEFINE_integer(
    "vocab_size", 1000,
    "The maximum number of vocabulary allowed, words beyond that will be counted as an "
    "unknown value")
tf.flags.DEFINE_string(
    "plain_vocab", "0,1,2,3,4,5,6,7,8,9",
    "The characters (comma separated) used for the plaintext vocabulary.")
tf.flags.DEFINE_string(
    "cipher_vocab", "10,11,12,13,14,15,16,17,18,19",
    "The characters (comma separated) used for the cipher text vocabulary.")
tf.flags.DEFINE_string(
    "distribution", None,
    "The distribution (comma separated) for each character of the vocabularies."
)
tf.flags.DEFINE_integer("sample_length", 100,
                        "The number of characters in each sample.")
tf.flags.DEFINE_string(
    "vigenere_key", "34",
    "The key for Vigenere cipher relates to the Vigenere table.")
tf.flags.DEFINE_bool("char_level", False,
                     "Whether the data is to be tokenized character-wise.")
tf.flags.DEFINE_integer(
    "shift_amount", 3,
    "The size of the shift for the shift cipher. -1 means random")
tf.flags.DEFINE_integer(
    "num_train", 50000,
    "The number of training samples to produce for each vocab.")
tf.flags.DEFINE_integer(
    "num_test", 5000, "The number of test samples to produce for each vocab.")
tf.flags.DEFINE_integer("num_shards", 1,
                        "The number of files to shard data into.")
tf.flags.DEFINE_bool("insert_unk", True,
                     "Insert <unk> if word is unknown due to vocab_size.")

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

# reserve 0 for pad
_CROP_AMOUNT = 1
_EXTRA_VOCAB_ITEMS = ["<pad>"]


class Layer():
  """A single layer for shift"""

  def __init__(self, vocab, shift):
    """Initialize shift layer

    Args:
      vocab (list of String): the vocabulary
      shift (Integer): the amount of shift apply to the alphabet. Positive number implies
            shift to the right, negative number implies shift to the left.
    """
    self.shift = shift
    alphabet = vocab
    shifted_alphabet = deque(alphabet)
    shifted_alphabet.rotate(shift)
    self.encrypt = dict(zip(alphabet, list(shifted_alphabet)))
    self.decrypt = dict(zip(list(shifted_alphabet), alphabet))

  def encrypt_character(self, character):
    return self.encrypt[character]

  def decrypt_character(self, character):
    return self.decrypt[character]


def generate_plaintext_random():
  """Generates samples of text from the provided vocabulary.
  Returns:
    train_indices (np.array of Integers): random integers generated for training.
      shape = [num_samples, length]
    test_indices (np.array of Integers): random integers generated for testing.
      shape = [num_samples, length]
    plain_vocab   (list of Integers): unique vocabularies.
  """
  plain_vocab = _EXTRA_VOCAB_ITEMS + FLAGS.plain_vocab.split(',')
  distribution = None if FLAGS.distribution is None else [
      float(x.strip()) for x in FLAGS.distribution.split(',')
  ]
  assert distribution is None or sum(distribution) == 1.0

  train_samples = FLAGS.num_train
  test_samples = FLAGS.num_test
  length = FLAGS.sample_length

  train_indices = np.random.choice(
      range(_CROP_AMOUNT, len(plain_vocab)), (train_samples, length),
      p=distribution)
  test_indices = np.random.choice(
      range(_CROP_AMOUNT, len(plain_vocab)), (test_samples, length),
      p=distribution)

  return train_indices, test_indices, plain_vocab


def generate_plaintext_corpus(character_level=False, insert_unk=True):
  """Load the corpus and divide it up into a training set and a test set before
  generating TFRecords
  Returns:
    train_indices (np.array of Integers): sentences generated from corpus for training.
      shape = [num_samples, length]
    test_indices (np.array of Integers): sentences generated from corpus for evaluation.
      shape = [num_samples, length]
    plain_vocab   (list of Integers): unique vocabularies in samples.
  """
  corpus = FLAGS.corpus
  # Sanity check provided corpus is a valid carpus available in nltk
  if nltk.download(corpus):
    corpus_method = getattr(nltk.corpus, corpus)
    word_frequency = determine_frequency(corpus_method, character_level)
    vocabulary = trim_vocab(word_frequency, FLAGS.vocab_size)
    plain_corpus, vocabulary = tokenize_corpus(corpus_method, vocabulary,
                                               _EXTRA_VOCAB_ITEMS,
                                               character_level, insert_unk)
    cutoff = int(FLAGS.percentage_training * len(plain_corpus))
    train_indices, test_indices = plain_corpus[:cutoff], plain_corpus[cutoff:]

    # truncate sentences
    for i in range(len(train_indices)):
      train_indices[i] = train_indices[i][:FLAGS.sample_length]
    for i in range(len(test_indices)):
      test_indices[i] = test_indices[i][:FLAGS.sample_length]

    return train_indices, test_indices, vocabulary
  else:
    raise ValueError(
        "The corpus you specified isn't available. Fix your corpus flag.")


def encipher_shift(plaintext, plain_vocab, shift):
  """Encrypt plain text with a single shift layer
  Args:
    plaintext (list of list of Strings): a list of plain text to encrypt.
    plain_vocab (list of Integer): unique vocabularies being used.
    shift (Integer): number of shift, shift to the right if shift is positive.
  Returns:
    ciphertext (list of Strings): encrypted plain text.
  """
  ciphertext = []
  cipher = Layer(range(_CROP_AMOUNT, len(plain_vocab)), shift)

  for i, sentence in enumerate(plaintext):
    cipher_sentence = []
    for j, character in enumerate(sentence):
      encrypted_char = cipher.encrypt_character(character)
      if FLAGS.separate_domains:
        encrypted_char += len(plain_vocab) - _CROP_AMOUNT
      cipher_sentence.append(encrypted_char)
    ciphertext.append(cipher_sentence)

  return ciphertext


def encipher_vigenere(plaintext, plain_vocab, key):
  """Encrypt plain text with given key
  Args:
    plaintext (list of list of Strings): a list of plain text to encrypt.
    plain_vocab (list of Integer): unique vocabularies being used.
    key (list of Integer): key to encrypt cipher using Vigenere table.
  Returns:
    ciphertext (list of Strings): encrypted plain text.
  """
  ciphertext = []
  # generate Vigenere table
  layers = []
  for i in range(len(plain_vocab)):
    layers.append(Layer(range(_CROP_AMOUNT, len(plain_vocab)), i))

  for i, sentence in enumerate(plaintext):
    cipher_sentence = []
    for j, character in enumerate(sentence):
      key_idx = key[j % len(key)]
      encrypted_char = layers[key_idx].encrypt_character(character)
      if FLAGS.separate_domains:
        encrypted_char += len(plain_vocab) - _CROP_AMOUNT
      cipher_sentence.append(encrypted_char)
    ciphertext.append(cipher_sentence)

  return ciphertext


def cipher_generator():
  """Generate text and cipher samples for conversion to TFRecords."""
  if FLAGS.corpus == "custom":
    train_plain, test_plain, plain_vocab = generate_plaintext_random()
  else:
    train_plain, test_plain, plain_vocab = generate_plaintext_corpus(
        character_level=FLAGS.char_level, insert_unk=FLAGS.insert_unk)

  if FLAGS.cipher == "shift":
    shift = FLAGS.shift_amount

    if shift == -1:
      shift = np.random.randint(1e5)

    train_cipher = encipher_shift(train_plain, plain_vocab, shift)
    test_cipher = encipher_shift(test_plain, plain_vocab, shift)
  elif FLAGS.cipher == "vigenere":
    key = [int(c) for c in FLAGS.vigenere_key]
    train_cipher = encipher_vigenere(train_plain, plain_vocab, key)
    test_cipher = encipher_vigenere(test_plain, plain_vocab, key)
  else:
    raise Exception("Unknown cipher %s" % FLAGS.cipher)

  save_vocab(plain_vocab, FLAGS.separate_domains, _CROP_AMOUNT,
             FLAGS.output_dir, FLAGS.vocab_filename)

  train_samples = zip(train_plain, train_cipher)
  test_samples = zip(test_plain, test_cipher)

  assert len(train_plain) == len(train_cipher)
  assert len(test_plain) == len(test_cipher)

  train_data = [{
      "X": plain_sample,
      "Y": cipher_sample,
      "set": [i % 2],
  } for i, (plain_sample, cipher_sample) in enumerate(train_samples)]
  test_data = [{
      "X": plain_sample,
      "Y": cipher_sample,
  } for plain_sample, cipher_sample in test_samples]

  # Random shuffle data
  np.random.shuffle(train_data)
  np.random.shuffle(test_data)

  generate_files(train_data, FLAGS.train_name, FLAGS.output_dir,
                 FLAGS.num_shards)
  generate_files(test_data, FLAGS.test_name, FLAGS.output_dir,
                 FLAGS.num_shards)


if __name__ == '__main__':
  cipher_generator()
