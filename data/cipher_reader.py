import tensorflow as tf

from .registry import register
from ..models.utils.model_utils import embed_inputs


@register("cipher")
def input_fn(data_sources, params, training):
  """Input function for integer cipher data."""

  if training:
    data_fields_to_features = {
        "X": tf.VarLenFeature(tf.int64),
        "Y": tf.VarLenFeature(tf.int64),
        "set": tf.FixedLenFeature([], tf.int64),
    }
  else:
    data_fields_to_features = {
        "X": tf.VarLenFeature(tf.int64),
        "Y": tf.VarLenFeature(tf.int64),
    }

  def _input_fn():
    """Input function compatible with Experiment API."""
    if training:
      filenames = tf.gfile.Glob(data_sources)
      filename_queue = tf.train.string_input_producer(filenames)
      _, serialized_example = tf.TFRecordReader().read(filename_queue)
      features = tf.parse_single_example(
          serialized=serialized_example, features=data_fields_to_features)

      plain_batch = tf.train.maybe_batch(
          features,
          tf.equal(features['set'], 0),
          params.batch_size,
          num_threads=4,
          capacity=5 * params.batch_size,
          dynamic_pad=True)
      cipher_batch = tf.train.maybe_batch(
          features,
          tf.equal(features['set'], 1),
          params.batch_size,
          num_threads=4,
          capacity=5 * params.batch_size,
          dynamic_pad=True)
    else:
      batch = tf.contrib.learn.read_batch_record_features(
          data_sources,
          params.batch_size,
          data_fields_to_features,
          randomize_input=False,
          num_epochs=None,
          queue_capacity=1e4,
          reader_num_threads=4 if training else 1)
      plain_batch, cipher_batch = batch, batch

    X = tf.sparse_tensor_to_dense(plain_batch["X"])
    Y = tf.sparse_tensor_to_dense(cipher_batch["Y"])
    X_ground_truth = tf.sparse_tensor_to_dense(cipher_batch["X"])
    Y_ground_truth = tf.sparse_tensor_to_dense(plain_batch["Y"])

    X = tf.pad(X, [[0, 0], [0, params.sample_length - tf.shape(X)[1]]])
    X.set_shape([params.batch_size, params.sample_length])
    Y = tf.pad(Y, [[0, 0], [0, params.sample_length - tf.shape(Y)[1]]])
    Y.set_shape([params.batch_size, params.sample_length])
    X_ground_truth = tf.pad(
        X_ground_truth,
        [[0, 0], [0, params.sample_length - tf.shape(X_ground_truth)[1]]])
    X_ground_truth.set_shape([params.batch_size, params.sample_length])
    Y_ground_truth = tf.pad(
        Y_ground_truth,
        [[0, 0], [0, params.sample_length - tf.shape(Y_ground_truth)[1]]])
    Y_ground_truth.set_shape([params.batch_size, params.sample_length])

    return {
        "X": X,
        "Y": Y,
        "X_ground_truth": X_ground_truth,
        "Y_ground_truth": Y_ground_truth
    }, {
        "X": tf.one_hot(X, depth=params.vocab_size),
        "Y": tf.one_hot(Y, depth=params.vocab_size),
        "X_ground_truth": tf.one_hot(X_ground_truth, depth=params.vocab_size),
        "Y_ground_truth": tf.one_hot(Y_ground_truth, depth=params.vocab_size)
    }

  return _input_fn
