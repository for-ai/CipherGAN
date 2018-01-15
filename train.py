import shutil
import os
import tensorflow as tf

from .hparams.registry import get_hparams
from .models.registry import _MODELS
from .data.registry import _INPUT_FNS, get_dataset
from .metrics.registry import get_metrics
from .train_utils.lr_schemes import get_lr
from .train_utils.vocab_utils import read_vocab

tf.flags.DEFINE_string("model", "cycle_gan", "Which model to use.")
tf.flags.DEFINE_string("data", "cipher", "Which data to use.")
tf.flags.DEFINE_string("hparam_sets", "cipher_default", "Which hparams to use.")
tf.flags.DEFINE_string("hparams", "", "Run-specific hparam settings to use.")
tf.flags.DEFINE_string("metrics", "xy_mse",
                       "Dash separated list of metrics to use.")
tf.flags.DEFINE_string("output_dir", "tmp/tf_run",
                       "The output directory.")
tf.flags.DEFINE_string("data_dir", "tmp/data", "The data directory.")
tf.flags.DEFINE_integer("train_steps", 1e4,
                        "Number of training steps to perform.")
tf.flags.DEFINE_integer("eval_steps", 1e2,
                        "Number of evaluation steps to perform.")
tf.flags.DEFINE_boolean("overwrite_output", False,
                        "Remove output_dir before running.")
tf.flags.DEFINE_string("train_name", "data-train*",
                       "The train dataset file name.")
tf.flags.DEFINE_string("test_name", "data-eval*", "The test dataset file name.")

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def _run_locally(train_steps, eval_steps):
  """Run training, evaluation and inference locally.

  Args:
    train_steps: An integer, number of steps to train.
    eval_steps: An integer, number of steps to evaluate.
  """
  hparams = get_hparams(FLAGS.hparam_sets)
  hparams = hparams.parse(FLAGS.hparams)
  hparams.total_steps = FLAGS.train_steps

  if "vocab_file" in hparams.values():
    hparams.vocab = read_vocab(hparams.vocab_file)
    hparams.vocab_size = len(hparams.vocab)
    hparams.vocab_size += int(hparams.vocab_size % 2 == 1)
    hparams.input_shape = [hparams.sample_length, hparams.vocab_size]

  output_dir = FLAGS.output_dir
  if os.path.exists(output_dir) and FLAGS.overwrite_output:
    shutil.rmtree(FLAGS.output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  def model_fn(features, labels, mode):
    lr = get_lr(hparams)
    return _MODELS[FLAGS.model](hparams, lr)(features, labels, mode)

  train_path, eval_path = get_dataset(FLAGS.data_dir, FLAGS.train_name,
                                      FLAGS.test_name)
  train_input_fn = _INPUT_FNS[FLAGS.data](train_path, hparams, training=True)
  eval_input_fn = _INPUT_FNS[FLAGS.data](eval_path, hparams, training=False)

  run_config = tf.contrib.learn.RunConfig()

  estimator = tf.contrib.learn.Estimator(
      model_fn=model_fn, model_dir=output_dir, config=run_config)

  eval_metrics = get_metrics(FLAGS.metrics, hparams)
  experiment = tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      eval_metrics=eval_metrics,
      train_steps=train_steps,
      eval_steps=eval_steps)
  experiment.train_and_evaluate()


def main(_):
  _run_locally(FLAGS.train_steps, FLAGS.eval_steps)


if __name__ == "__main__":
  tf.app.run()
