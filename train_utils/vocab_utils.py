import codecs
import os
import re

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def read_vocab(filename):
  vocab = dict()
  filepath = os.path.join(FLAGS.data_dir, filename)
  with tf.gfile.FastGFile(filepath) as vocab_file:
    for line in vocab_file:
      match = re.search(r'(\d+)\. (.*)\s*', line)
      vocab[int(match.group(1))] = match.group(2)

  pairs = sorted(vocab.items(), key=lambda a: a[0])
  indices = [a for a, b in pairs]
  words = [b for a, b in pairs]
  assert indices == list(range(len(indices)))
  return words


def read_corpus(filename, encoding="utf-8"):
  """Reads a textfile corpus.

  Args:
    filename (String) : The path to the file containing the corpus.
    encoding (String ): The encoding on the file. (default: {"utf-8"})

  Returns:
    corpus (np.ndarray) : An array of integers representing the corpus.
    mappings (dict) : A dictionary of mappings where the keys are characters
                      and the values are the corresponding integers.
  """
  corpus = None
  mappings = None
  with codecs.open(filename, "r", encoding) as file:
    data = file.read()
    chars = set(data)
    mappings = {c: i for i, c in enumerate(chars)}
    corpus = np.array([mappings[ch] for ch in data])
  return corpus, mappings
