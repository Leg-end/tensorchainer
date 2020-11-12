from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import inspect
from tensorflow.python.ops import math_ops
from tensorflow import dtypes
import os
import re
import collections
import numpy as np
import time

try:
    import ipykernel
except ImportError:
    print("Installing `ipykernel`")
    res = os.system("pip install ipykernel")
    if res != 0:
        print("Install `ipykernel` failed")
    else:
        print("Successfully install `ipykernel`")
try:
    import posix
except ImportError:
    print("Installing `posix`")
    res = os.system("pip install posix")
    if res != 0:
        print("Install `posix` failed")
    else:
        print("Successfully install `posix`")

__all__ = ["to_list", "to_tuple",
           "unpack_singleton", "object_list_uid",
           "valid_value", "has_arg", "list_dirs",
           "list_files", "pre_dir", "find_str",
           "middle_merge", "check_mutex",
           "may_div_zero", "arg_count",
           "validate_kwargs", "valid_range",
           "slice_arrays", "ProgressBar"]


def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def to_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return x,


def unpack_singleton(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    else:
        return x


def validate_kwargs(kwargs, allowed_kwargs):
    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError('Keyword argument not understood:', kwarg)


def may_div_zero(x, y):
    if x.dtype is not dtypes.float32:
        x = dtypes.cast(x, dtype=dtypes.float32)
    if y.dtype is not dtypes.float32:
        y = dtypes.cast(y, dtype=dtypes.float32)
    return math_ops.greater(y, 0.) * math_ops.divide(
        x, math_ops.maximum(1e-5, y))


def object_list_uid(object_list):
    object_list = to_list(object_list)
    return ','.join([str(id(x)) for x in object_list])


def valid_range(value, legal_range):
    assert len(legal_range) == 2
    if not legal_range[0] < value < legal_range[1]:
        raise ValueError("Expect argument in range {},"
                         " but received {}".format(
                          str(legal_range), str(value)))
    return value


def valid_value(value, legal_values):
    assert hasattr(legal_values, '__iter__')
    if value not in legal_values:
        raise ValueError("Expect argument in {}, but"
                         " received {}".format(
                          str(value), str(legal_values)))
    return value


def check_mutex(*params, names):
    true_num = 0
    for param in params:
        if param is not None:
            true_num += 1
        if true_num > 1:
            raise ValueError("Expect only one effective param in {},"
                             "but receive multiple effective params,"
                             "this is ambiguous".format(names))
    if true_num == 0:
        raise ValueError("Expect only one effective param in {},"
                         "but no effective param received".format(names))


def has_arg(func, name, accept_all=False):
    """
    Checks if a callable accepts a given keyword argument.
    :param func: callable
    :param name: keyword argument's name
    :param accept_all: True if no keyword named 'name',
     but the function has **kwargs argument, True will be returned
    :return: bool whether has arg named 'name' in func
    """
    if sys.version_info < (3,):
        arg_spec = inspect.getfullargspec(func)
        if accept_all and arg_spec.varkw is not None:
            return True
        return name in arg_spec.args
    elif sys.version_info < (3, 3):
        arg_spec = inspect.getfullargspec(func)
        if accept_all and arg_spec.varkw is not None:
            return True
        return name in arg_spec.args or name in arg_spec.kwonlyargs
    else:
        signature = inspect.signature(func)
        parameter = signature.parameters.get(name)
        if parameter is None:
            if accept_all:
                for param in signature.parameters.values():
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        return True
            return False
        return parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY)


def arg_count(func):
    try:
        init_code = func.__func__.__code__
    except AssertionError:
        raise ValueError("Cannot determine args to func %s" % func)
    attr_names = init_code.co_varnames[:init_code.co_argcount]
    if 'self' in attr_names:
        attr_names = attr_names[1:]
    return len(attr_names)


def list_files(user_dir, only_name=False, parent_dir=None, suffixes=None):
    for root, dirs, files in os.walk(user_dir):
        if dirs:
            for directory in dirs:
                if parent_dir:
                    parent_dir = os.path.join(parent_dir, root)
                list_files(directory, parent_dir=parent_dir, suffixes=suffixes)
        if files:
            if only_name:
                if suffixes:
                    for file in filter(lambda x: x[x.rfind('.') + 1:] in suffixes, files):
                        yield file
                else:
                    for file in files:
                        yield file
            else:
                for file in files:
                    if suffixes and file[file.rfind('.') + 1:] not in suffixes:
                        continue
                    if parent_dir:
                        path = os.path.join(parent_dir, root, file)
                    else:
                        path = os.path.join(root, file)
                    path = path.replace('\\', '/')
                    yield path


def list_dirs(user_dir, parent_dir=None):
    for root, dirs, files in os.walk(user_dir):
        if dirs:
            for directory in dirs:
                if parent_dir:
                    parent_dir = os.path.join(parent_dir, root)
                list_dirs(directory, parent_dir)
        else:
            yield root


def pre_dir(path):
    path = path.replace('\\', '/')
    return path[:path.rfind('/')]


def find_str(value, pattern, num=None, reverse=False, start=True):
    indices = list(re.finditer(pattern, value))
    if reverse:
        indices = list(reversed(indices))
    if len(indices) == 0:
        return -1
    if start:
        indices = [i.start() for i in indices]
    else:
        indices = [i.end() for i in indices]
    if num >= len(indices):
        return -1
    elif num >= 0:
        return indices[num]
    else:
        return indices


def middle_merge(x: str, y: str, pattern='/'):
    """
    Merge ?/?/?/a/b/c/, a/b/c/!/!/!/ to ?/?/?/a/b/c/!/!/!/
    pattern specified with '/'
    """
    x = x.split(pattern)
    y = y.split(pattern)
    x_tmp = x.copy()
    for i, v in enumerate(y):
        if v in x_tmp:
            y[i] = ''
            x_tmp.pop(x_tmp.index(v))
        else:
            break
    result = list(filter(lambda z: z != '', x + y))
    return pattern.join(result) + pattern


def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            return [None if x is None else x[start: stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start: stop]
        else:
            return [None]


class ProgressBar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules or
                                 'PYCHARM_HOSTED' in os.environ)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
