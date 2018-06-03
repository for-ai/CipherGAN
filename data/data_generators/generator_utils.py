import operator
import os
import numpy as np
import tensorflow as tf


def to_example(dictionary):
  features = {}
  for k, v in dictionary.items():
    if len(v) == 0:
      raise Exception("Empty field: %s" % str((k, v)))
    if isinstance(v[0], (int, np.int8, np.int32, np.int64)):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], (float, np.float32)):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], (str, bytes)):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise Exception("Unsupported type: %s" % type(v[0]))
  return tf.train.Example(features=tf.train.Features(feature=features))


def generate_files(generator,
                   output_name,
                   output_dir,
                   num_shards,
                   max_cases=None):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  writers = []
  for shard in range(num_shards):
    output_filename = "%s-%dof%d" % (output_name, shard + 1, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    writers.append(tf.python_io.TFRecordWriter(output_file))

  counter, shard = 0, 0
  for case in generator:
    if counter % 100 == 0:
      tf.logging.info("Processed %d examples..." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    sequence_example = to_example(case)
    writers[shard].write(sequence_example.SerializeToString())
    shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()


def save_vocab(vocab, separate_domains, crop, output_dir, vocab_filename):
  """Save vocabulary to file
  Args:
    vocab (list of String): the list of vocabularies.
    separate_domains (Boolean): indicate the plaintext and cipher are in separate domain.
    crop: (Integer): the amount of crop applied.
    output_dir (String): path to output directory.
    vocab_filename (String): output vocabulary filename
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  mapping_fd = open(os.path.join(output_dir, vocab_filename), 'w')
  for i, word in enumerate(vocab):
    mapping_fd.write("{0}. {1}\n".format(i, word))
    if separate_domains and i >= crop:
      mapping_fd.write("{0}. {1}\n".format(i + len(vocab) - crop, word))
  mapping_fd.close()


def string2index(sentences, vocab):
  """Convert string to its corresponding index
  i.e. A -> 0, B -> 1 ... for vocab [A,B,...]
  Args:
    sentences (np.array of String): list of String to convert.
     shape = [num_samples, length]
    vocab (list of String): list of vocabulary
  Returns:
      index (np.array of Integer): list of Integer after conversion
        shape = [num_samples, length]
  """
  alphabet_index = list(range(len(vocab)))
  mapping = dict(zip(vocab, alphabet_index))
  index = []
  for i in range(len(sentences)):
    sentence = []
    for j in range(len(sentences[i])):
      sentence.append(mapping[sentences[i, j]])
    index.append(sentence)
  return index


def trim_vocab(word_frequency, vocab_size):
  """Given the max vocab size n, trim the word_frequency dictionary to only contain the
  top n occurring words
  Args:
    word_frequency (Dictionary): dictionary of word, frequency pairs.
    vocab_size (Integer): the maximum number of vocabulary allowed.
  Returns:
    retval (Dictionary): dictionary containing the top n occurring words as keys
  """
  sorted_vocab = sorted(
      word_frequency.items(), key=operator.itemgetter(1), reverse=True)
  max_count = min(len(word_frequency), vocab_size)
  retval = [k for k, _ in sorted_vocab[:max_count]]
  return retval


def determine_frequency(corpus, character_level):
  """Go through corpus and determine frequency of each individual word
  Args:
    corpus (CategorizedTaggedCorpusReader): corpus object for the corpus being used
  Returns:
    unique_word_count (Dictionary): dictionary of word keys and corresponding frequency
      value
  """
  unique_word_count = dict()
  lengths = []
  for sentence in corpus.sents():
    if not character_level:
      lengths.append(len(sentence))
    else:
      lengths.append(sum(len(word) for word in sentence))
    for word in sentence:
      if character_level:
        for character in word:
          if not character.lower() in unique_word_count:
            unique_word_count[character.lower()] = 1
          else:
            unique_word_count[character.lower()] += 1
      else:
        if not word.lower() in unique_word_count:
          unique_word_count[word.lower()] = 1
        else:
          unique_word_count[word.lower()] += 1
  print("Average sentence length: %d" % (sum(lengths) / len(lengths)))
  print("Max sentence length: %d" % (max(lengths)))
  print("Min sentence length: %d" % (min(lengths)))
  return unique_word_count


def tokenize_corpus(corpus, vocabulary, additional_items, character_level,
                    insert_unk):
  """Translate string words into int ids
  Args:
    corpus (CategorizedTaggedCorpusReader): corpus object for the corpus being used.
    vocabulary (Dictionary): vocabulary being used. Also write vocab mapping to file.
  Returns:
    tokenized_corpus (Dictionary): tokenized corpus.
    ordered_vocab (list of String): ordered vocabulary
  """
  unique_words = dict()
  vocab = dict()
  tokenized_corpus = []

  shift = len(additional_items)
  unique_count = shift + 1
  for i, item in enumerate(additional_items):
    vocab[i] = item
  vocab[shift] = "<unk>"
  for sentence in corpus.sents():
    tokenized_sentence = []
    for word in sentence:
      word = word.lower()
      if character_level:
        for character in word:
          if not character in unique_words:
            if character in vocabulary:
              unique_words[character] = unique_count
              vocab[unique_count] = character
              unique_count += 1
            else:
              # shift reserved for other unknown words
              unique_words[character] = shift

          if unique_words[character] == shift and insert_unk:
            tokenized_sentence.append(unique_words[character])
          elif unique_words[character] != shift:
            tokenized_sentence.append(unique_words[character])
      else:
        if not word in unique_words:
          if word in vocabulary:
            unique_words[word] = unique_count
            vocab[unique_count] = word
            unique_count += 1
          else:
            # shift reserved for other unknown words
            unique_words[word] = shift

        if unique_words[word] == shift and insert_unk:
          tokenized_sentence.append(unique_words[word])
        elif unique_words[word] != shift:
          tokenized_sentence.append(unique_words[word])

    if len(tokenized_sentence) > 0:
      tokenized_corpus.append(tokenized_sentence)
  sorted_vocab = sorted(vocab.items(), key=lambda a: a[0])
  ordered_vocab = [v for _, v in sorted_vocab]
  return tokenized_corpus, ordered_vocab
