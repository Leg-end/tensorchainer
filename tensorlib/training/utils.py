import numpy as np
from tensorflow.python import ops
from tensorlib.training.sessions.core import Compiler, SaverTrigger


def is_tensor(x):
    return isinstance(x, getattr(ops, '_TensorLike')) or ops.is_dense_tensor_like(x)


def normalize_single_array(x):
    if x is None:
        return None
    elif is_tensor(x):
        shape = x.get_shape()
        if shape is None or shape[0] is None:
            raise ValueError(
                'When feeding symbolic tensors to a model, we expect the'
                'tensors to have a static batch size. '
                'Got tensor with shape: %s' % str(shape))
        return x
    elif x.ndim == 1:
        x = np.expand_dims(x, 1)
    return x


def verify_and_normalize_data(data,
                              names,
                              shapes=None):
    if not names:
        if data is not None and hasattr(data, '__len__') and len(data):
            raise ValueError('Error when checking model: expected no data, but got:', data)
        return []
    if data is None:
        return [None] * len(names)
    if isinstance(data, list):
        if isinstance(data[0], list):
            data = [np.asarray(d) for d in data]
        elif len(names) == 1 and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
    else:
        data = [data]
    # data = [normalize_single_array(x) for x in data]
    if len(data) != len(names):
        if data and hasattr(data[0], 'shape'):
            raise ValueError(
                'Error when checking model: the list of Numpy arrays that you are passing to '
                'your model is not the size the model expected. Expected to see ' + str(len(names)) +
                ' array(s), but instead got the following list of ' + str(len(data)) + ' arrays: ' +
                str(data)[:min(200, len(data))] + '...')
        elif len(names) > 1:
            raise ValueError(
                'Error when checking model : you are passing a list as input to your model, '
                'but the model expects a list of ' + str(len(names)) + ' Numpy arrays instead. '
                'The list you passed was: ' + str(data)[:min(200, len(data))])
        elif len(data) == 1 and not hasattr(data[0], 'shape'):
            raise TypeError(
                'Error when checking model: data should be a Numpy array, or list/dict of '
                'Numpy arrays. Found: ' + str(data)[:min(200, len(data))] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]

    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None and not is_tensor(data[i]):
                data_shape = data[i].shape
                shape = shapes[i]
                if data[i].ndim != len(shape):
                    raise ValueError(
                        'Error when checking : expected ' + names[i] + ' to have ' +
                        str(len(shape)) + ' dimensions, but got array with shape ' + str(data_shape))
                data_shape = data_shape[1:]
                shape = shape[1:]
                for dim, ref_dim in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim:
                        raise ValueError(
                            'Error when checking : expected ' + names[i] + ' to have shape ' +
                            str(shape) + ' but got array with shape ' + str(data_shape))
    return data


def validate_compiler(compiler):
    compiler = compiler or Compiler()
    if not isinstance(compiler, compiler):
        raise TypeError('compiler must be Compiler. Given: {}'.format(compiler))
    return compiler


def validate_saver_triggers(saver_triggers):
    triggers = list(saver_triggers or [])
    for t in triggers:
        if not isinstance(t, SaverTrigger):
            raise TypeError(
                'saver_triggers must be a list of SaverTrigger,'
                'given: {}'.format(t))
    return triggers


def check_num_samples(samples,
                      batch_size=None,
                      steps=None):
    if steps is not None and batch_size is not None:
        raise ValueError('When `steps` is set, the `batch_size` must be None')
    if not samples or any(is_tensor(x) for x in samples):
        if steps is None:
            raise ValueError("When samples from symbolic tensors(e.g. Dataset), argument"
                             " `steps` must be specified instead of batch_size, cause"
                             " symbolic tensors are expected to produce batches of data")
        return None
    if hasattr(samples[0], 'shape'):
        return int(samples[0].shape[0])
    return None


def check_array_length_consistency(inputs, targets):
    """Checks if batch axes are the same for numpy arrays.

    # Arguments
        inputs: list of Numpy arrays of inputs.
        targets: list of Numpy arrays of targets.

    # Raises
        ValueError: in case of incorrectly formatted data.
    """
    def set_of_lengths(x):
        # return a set with the variation between
        # different shapes, with None => 0
        if x is None:
            return {0}
        else:
            return set([0 if y is None else int(y.shape[0]) for y in x])

    set_x = set_of_lengths(inputs)
    set_y = set_of_lengths(targets)
    if len(set_x) > 1:
        raise ValueError('All input arrays (x) should have '
                         'the same number of samples. Got array shapes: ' +
                         str([x.shape for x in inputs]))
    if len(set_y) > 1:
        raise ValueError('All target arrays (y) should have '
                         'the same number of samples. Got array shapes: ' +
                         str([y.shape for y in targets]))
    if set_x and set_y and list(set_x)[0] != list(set_y)[0]:
        raise ValueError('Input arrays should have '
                         'the same number of samples as target arrays. '
                         'Found ' + str(list(set_x)[0]) + ' input samples '
                         'and ' + str(list(set_y)[0]) + ' target samples.')


def shuffle(indices, batch_size=None):
    if batch_size:
        batch_count = len(indices) // batch_size
        last_batch = indices[batch_count * batch_size:]
        indices = indices[:batch_count * batch_size]
        indices = indices.reshape((batch_count, batch_size))
        np.random.shuffle(indices)
        indices = indices.flatten()
        indices = np.append(indices, last_batch)
    else:
        np.random.shuffle(indices)
    return indices


def make_batches(size, batch_size):
    num_batches = (size + batch_size - 1) // batch_size
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]
