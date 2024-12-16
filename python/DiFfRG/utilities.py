import sys
import uuid
import glob

def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

def get_all_sims(folder) -> list:
    return glob.glob(folder + "*.pvd")