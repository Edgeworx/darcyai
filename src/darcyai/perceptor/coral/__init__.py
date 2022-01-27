"""Version information for darcyai-coral Python APIs."""

__version__ = "0.2.5"

from importlib import import_module

edgetpu = import_module("pycoral.utils.edgetpu")
dataset = import_module("pycoral.utils.dataset")
