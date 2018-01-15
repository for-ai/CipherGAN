import os

_INPUT_FNS = dict()


def register(name):

  def add_to_dict(fn):
    global _INPUT_FNS
    _INPUT_FNS[name] = fn
    return fn

  return add_to_dict


def get_dataset(data_dir, train_name, test_name):
  return [
      os.path.join(data_dir, data_path)
      for data_path in [train_name, test_name]
  ]
