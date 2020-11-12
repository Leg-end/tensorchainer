from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from distutils.version import LooseVersion


if 'TENSORLIB_PACKAGE_BUILDING' not in os.environ:
    try:
        import tensorflow
    except Exception as e:
        raise ImportError(
            "Tensorflow is not installed, please install it with one of the following commands:\n"
            "\t`pip install --upgrade tensorflow` or\n"
            "\t`pip install --upgrade tensorflow-gpu`"
        )
    if LooseVersion(tensorflow.__version__) < LooseVersion("1.9.0"):
        raise RuntimeError(
            "TensorLib does not support Tensorflow version older than 1.14.0\n"
            "Please update Tensorflow with:\n"
            "\t`pip install --upgrade tensorflow` or\n"
            "\t`pip install --upgrade tensorflow-gpu`"
        )

from tensorlib import utils
from tensorlib import initializers
from tensorlib import regularizers
from tensorlib import activation_ops
from tensorlib import layers
from tensorlib import research
from tensorlib import hooks
from tensorlib import contrib
from tensorlib import data
from tensorlib import training
from tensorlib.engine import arg_scope
from tensorlib.engine import Input
from tensorlib.engine import Network
from tensorlib.engine import graph_scope
from tensorlib.engine import Layer
from tensorlib.engine import LayerList
from tensorlib.engine import Sequential
from tensorlib.engine import get_session
from tensorlib.engine import clear_session

from collections import OrderedDict
import threading

_thread_local = threading.local()


def _get_hooks():
    try:
        hooks = _thread_local.hooks
    except AttributeError:
        hooks = OrderedDict()
        _thread_local.hooks = hooks
    return hooks


del absolute_import
del division
del print_function
