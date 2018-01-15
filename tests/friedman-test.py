from __future__ import division
import re
import nltk
from ..data.data_generators.cipher_generator import Layer

_KEYS = [3, 4, 5]
_VOCAB = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

def coincidence_rate(text):
  """ Return the coincidence rate of the given text
  Args:
    text (string): the text to get measured
  Returns:
    the coincidence rate
  """
  ko = 0
  # measure the frequency of each letter in the cipher text
  for letter in _VOCAB:
    count = text.count(letter)
    ko = ko + (count * (count - 1))
  return ko / (len(text) * (len(text) - 1))


def key_length_estimate(cipher):
  """ Return the estimated key length base on Friedman test
  Args:
    cipher (list of string): list of cipher text
  Returns:
    the estimated key length of the cipher text
  """
  kp = 1 / len(_VOCAB)
  kr = 0.067
  ko = coincidence_rate(cipher)
  return (kp - kr) / (ko - kr)


def generate_corpus():
  """Load brown corpus, convert list of unicode to string and remove
  punctuations
  Returns:
    list of sentences
  """
  if nltk.download('brown'):
    sentences = []
    corpus = getattr(nltk.corpus, 'brown')
    for sentence in corpus.sents():
      # convert unicode to string and list of strings to string
      encoded = ''.join([word.encode('UTF8').upper() for word in sentence])
      encoded = re.sub('[^A-Za-z]+', '', encoded)
      if len(encoded) > 1: sentences.append(encoded)
    return sentences
  else:
    raise ValueError("Corpus isn't available.")


def encipher(sentences, shifts):
  """
  Encipher the provided sentences with _SHIFT shifts
  Args:
    sentences (list of string): list of sentences to get encipher
    shifts (list of Layer): encipher
  Returns
    the enciphered sentences
  """
  ciphers = []
  for sentence in sentences:
    cipher = ""
    for i in range(len(sentence)):
      shift = shifts[i % len(shifts)]
      cipher += shift.encrypt_character(sentence[i])
    ciphers.append(cipher)
  return ciphers


def friedman_test():
  """ Return a list of estimated key length using Friedman test
  """
  corpus = generate_corpus()
  shifts = []
  for key in _KEYS:
    shifts.append(Layer(_VOCAB, key))
  ciphers = encipher(corpus[len(corpus)//2:], shifts)
  keys_length_estimation = []
  for cipher in ciphers:
    keys_length_estimation.append(key_length_estimate(cipher))
  return keys_length_estimation


if __name__ == '__main__':
  keys_length_estimation = friedman_test()
  print("Measured " + str(len(keys_length_estimation)) +
        " ciphers with key " + str(_KEYS))
  print("The average estimated key length using Friedman test is " +
        str(sum(keys_length_estimation) / len(keys_length_estimation)))
