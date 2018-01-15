"""
Preform simple frequency analysis on shift cipher
Compare the n-grams of first half of brown corpus with the second
half of brown corpus that is encrypted with shift cipher
"""
from __future__ import division
import nltk
from ..data.data_generators.cipher_generator import Layer

_KEYS = [3, 4, 5]
_PER_KEY = True


def get_ngram(sentences, n):
  """
  Return a dictionary with keys of the ngram and values of the
  count of how many times they appear in the provided sentences
  Args:
    sentences (list string): the sentences to be analyzed
    n (integer): the n contiguous sequence character
    """
  dictionary = {}
  for sentence in sentences:
    for i in range(len(sentence) - n + 1):
      ngram = sentence[i: i + n]
      if ngram in dictionary:
        dictionary[ngram] += 1
      else:
        dictionary[ngram] = 1
  return dictionary


def analysis_ngrams(sentences):
  """
  Args:
      sentences (list of string): sentences to be analyzed
  Returns:
      list of list of dictionaries of ngram analysis in the
      order of unigram, per-key unigram, bigram, and trigram
  """
  unigram = get_ngram(sentences, 1)
  bigram = get_ngram(sentences, 2)
  trigram = get_ngram(sentences, 3)

  key_unigram = []
  # if per-key unigram analysis is enabled, then analysis each sentence with
  # the key index
  if _PER_KEY:
    for i in range(len(_KEYS)):
      per_key_sentences = []
      for sentence in sentences:
        per_key_sentences.append(sentence[i::3])
      key_unigram.append(get_ngram(per_key_sentences, 1))

  return [[unigram], key_unigram, [bigram], [trigram]]


def generate_corpus():
  """Load brown corpus, convert list of unicode to string and remove
  punctuations
  Returns:
    list of sentences and a list of vocabs
  """
  if nltk.download('brown'):
    sentences = []
    vocab = []
    corpus = getattr(nltk.corpus, 'brown')
    for sentence in corpus.sents():
      # convert unicode to string and list of strings to string
      encoded = ''.join([word.encode('UTF8').upper() for word in sentence])
      for char in encoded:
        if char not in vocab:
          vocab.append(char)
      sentences.append(encoded)
    vocab.sort()
    return sentences, vocab
  else:
    raise ValueError("Corpus isn't available.")


def decipher(cipher, shifts):
  """
  Decipher the provided cipher with _SHIFT shifts
  Args:
    cipher (string): string to be decipher
    shifts (list of Layer): encipher
  Returns
    the plaintext
  """
  plaintext = ""
  for i in range(len(cipher)):
    shift = shifts[i % len(shifts)]
    plaintext += shift.decrypt_character(cipher[i])
  return plaintext


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


def frequency_anaylsis(plaintext_ngrams, cipher_ngrams, shifts):
  """
  Compare n-gram to the truth value in the cipher text. If the n-gram 
  prediction is correct,  then added the amount matched characters to the 
  counter, and divide by the length of the cipher at the end to calculate 
  the accuracy. Return the frequency analysis with the given plaintext 
  n-grams and cipher n-grams
  Args:
    plaintext_ngrams: (list of dictionary) list of dictionary of frequency 
      analyzed plain text
    cipher_ngrams: (list of dictionary) list of dictionary of frequency analyzed
      cipher text
    shifts (list of Layer): encipher
  Returns
    the accuracy of each n-gram
  """
  result = []
  # get each n-gram
  for ngram in range(len(plaintext_ngrams)):
    plaintext_ngram = plaintext_ngrams[ngram]
    cipher_ngram = cipher_ngrams[ngram]
    for key in range(len(plaintext_ngram)):
      # get a list of keys from plaintext and cipher sorted by their values
      plaintext_sorted = sorted(plaintext_ngram[key],
                                key=plaintext_ngram[key].__getitem__, 
                                reverse=True)
      cipher_sorted = sorted(cipher_ngram[key],
                             key=cipher_ngram[key].__getitem__, reverse=True)

      if len(cipher_ngram) == len(_KEYS):
        print("measuring accuracy for key " + str(key) + " in uni-gram...")
      else:
        print("measuring accuracy for " + str(ngram + 1) + "-gram with "
              + str(min(len(plaintext_sorted), len(cipher_sorted))) 
              + " results...")
      print("  most frequent plaintext: " + str(plaintext_sorted[:5]))
      print("  most frequent cipher: " + str(cipher_sorted[:5]) + "\n")

      # compare n-gram with truth value
      total = sum(list(cipher_ngram[key].values()))
      match = 0
      for value in range(min(len(plaintext_sorted), len(cipher_sorted))):
        # use the particular key when analogising per-key unigram
        if len(cipher_ngram) == len(_KEYS):
          truth = decipher(cipher_sorted[value], [shifts[key]])
        else:
          truth = decipher(cipher_sorted[value], shifts)
        prediction = plaintext_sorted[value]
        if truth == prediction:
          match += cipher_ngram[key][cipher_sorted[value]]
      # add accuracy to result
      result.append((match / total) * 100)
  return result


def analysis():
  """
  Analysis plain text and cipher text
  Returns:
    unigram, per key unigram, bi-gram and trigram accuracy of the plain text 
    and cipher text
  """
  corpus, vocab = generate_corpus()
  print("Encrypting with key: " + str(_KEYS))
  shifts = []
  vocab = ''.join(vocab)
  for key in _KEYS:
    shifts.append(Layer(vocab, key))

  print("analysing Brown corpus n-grams...")
  plaintext_ngrams = analysis_ngrams(corpus[:len(corpus) // 2])

  print("analysing shift cipher n-grams...\n")
  cipher = encipher(corpus[len(corpus) // 2:], shifts)
  cipher_ngrams = analysis_ngrams(cipher)

  return frequency_anaylsis(plaintext_ngrams, cipher_ngrams, shifts)


if __name__ == '__main__':
  results = analysis()
  ngrams = ['unigram', 'bigram', 'trigram']
  if _PER_KEY:
    for key in range(len(_KEYS)):
      ngrams.insert(1 + key, '  unigram key-' + str(key))
  for i in range(len(results)):
    print(ngrams[i] + " has accuracy " + str(results[i]) + "%")
