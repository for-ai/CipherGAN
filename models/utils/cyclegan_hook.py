from collections import deque

import numpy as np
import tensorflow as tf


class CycleGANHook(tf.train.SessionRunHook):

  def __init__(self, fetches, placeholders, params):
    super(CycleGANHook, self).__init__()
    self.fetches = fetches
    self.placeholders = placeholders
    self.params = params
    self.past_Xs, self.past_Ys = (deque([np.zeros(params.input_shape)]),
                                  deque([np.zeros(params.input_shape)]))

  def after_run(self, run_context, run_values):
    super(CycleGANHook, self).after_run(run_context, run_values)

    latest_X = run_values.results["X"]
    latest_Y = run_values.results["Y"]

    if len(self.past_Xs) + len(latest_X) >= self.params.past_count:
      for _ in range(
          len(self.past_Xs) + len(latest_X) - self.params.past_count):
        if len(self.past_Xs) == 0:
          break
        self.past_Xs.popleft()
    if len(self.past_Ys) + len(latest_Y) >= self.params.past_count:
      for _ in range(
          len(self.past_Ys) + len(latest_Y) - self.params.past_count):
        if len(self.past_Ys) == 0:
          break
        self.past_Ys.popleft()

    for x, y in zip(latest_X, latest_Y):
      if len(self.past_Xs) == self.params.past_count or len(
          self.past_Ys) == self.params.past_count:
        break
      self.past_Xs.append(x)
      self.past_Ys.append(y)

    assert len(self.past_Xs) <= self.params.past_count
    assert len(self.past_Ys) <= self.params.past_count

  def before_run(self, run_context):
    super(CycleGANHook, self).before_run(run_context)
    feed_dict = {
        self.placeholders["X"]: np.array(self.past_Xs),
        self.placeholders["Y"]: np.array(self.past_Ys)
    }
    return tf.train.SessionRunArgs(self.fetches, feed_dict)
