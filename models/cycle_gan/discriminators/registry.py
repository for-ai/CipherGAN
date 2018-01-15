_DISCRIMINATORS = dict()


def register(name):

  def add_to_dict(fn):
    global _DISCRIMINATORS
    _DISCRIMINATORS[name] = fn
    return fn

  return add_to_dict
