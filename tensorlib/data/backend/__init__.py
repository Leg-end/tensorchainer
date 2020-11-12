import sys

_BACKEND = 'tensorflow'

# Import backend functions.
if _BACKEND == 'opencv':
    sys.stderr.write('Using OpenCV backend\n')
    from .cv_backend import *
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using Tensorflow backend.\n')
    from .cv_backend import *


def get_backend():
    return _BACKEND