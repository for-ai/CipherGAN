_TRANSFORMATIONS = dict()


def register(name):

  def add_to_dict(fn):
    global _TRANSFORMATIONS
    _TRANSFORMATIONS[name] = fn
    return fn

  return add_to_dict
