import h5py
from tensorlib.engine.base_layer import Layer
from tensorlib.engine.base_lib import batch_set_value, get_session
from distutils.version import LooseVersion
from inspect import isgeneratorfunction
from tensorflow import logging
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops as fops
from tensorflow.python import pywrap_tensorflow


def _check_version():
    try:
        import tensorflow
    except Exception:
        raise ImportError("TensorFlow is not installed !")
    return LooseVersion(tensorflow.__version__)


def get_weight_value(weights):
    if isgeneratorfunction(weights):
        var_list = list(weights)
    elif not isinstance(weights, list):
        var_list = [weights]
    else:
        var_list = weights
    if _check_version() < LooseVersion("2.0.0"):
        values = get_session().run(var_list)
        return values
    else:
        return [v.numpy() for v in var_list]


def save_hdf5_weights(filepath, lib_model):
    with h5py.File(filepath, "w")as f:
        _save_weights_to_hdf5_group(f, lib_model.layers())


def _save_weights_to_hdf5_group(f, layers):
    f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in layers]

    for layer in layers:
        g = f.create_group(layer.name)
        if isinstance(layer, Layer):
            weight_values = []
            weight_names = []
            if layer.weights is not None:
                weight_values = get_weight_value(layer.weights)
                weight_names = [w.name.encode('utf8') for w in layer.weights]
            g.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                val_dataset = g.create_dataset(name, val.shape, dtype=val.dtype)
                if not val.shape:
                    val_dataset[()] = val
                else:
                    val_dataset[:] = val
        else:
            raise Exception("Only layer can be saved into hdf5")
        

def is_hdf5_format(filepath):
    return (filepath.endswith('.h5') or
            filepath.endswith('.hdf5'))


def load_hdf5_weights(filepath,
                      lib_model,
                      allow_skip=False):
    import warnings
    f = h5py.File(filepath, 'r')
    try:
        layer_names = [n.decode('utf8') for n in f.attrs["layer_names"]]
    except Exception:
        raise NameError(
            "The loaded hdf5 file needs to have 'layer_names' as attributes.")
    model_index = dict()
    count = 0
    for layer in lib_model.layers():
        model_index[layer.name] = layer
        count += 1
    if count != len(layer_names):
        warnings.warn("Trying to load a saved file with %d layers into a"
                      " model with %d layers".format(len(layer_names), count),
                      stacklevel=2)
    for name in model_index.keys():
        if name not in layer_names:
            warnings.warn("layer named '%s' not found in loaded hdf5 file, It will be skipped." % name,
                          stacklevel=2)
    weight_tuples = _load_weights_from_hdf5_group(f, lib_model.layers(), allow_skip=allow_skip)
    if _check_version() < LooseVersion("2.0.0"):
        batch_set_value(weight_tuples)
    else:
        for var, value in weight_tuples:
            var.assign(value)
    f.close()


def _load_weights_from_hdf5_group(f, layers, allow_skip=False):
    import numpy as np
    import warnings
    layer_names = [n.decode('utf8') for n in f.attrs["layer_names"]]
    layer_index = {layer.name: layer for layer in layers}
    weight_tuples = []
    for idx, name in enumerate(layer_names):
        if name not in layer_index.keys():
            if allow_skip:
                warnings.warn("Layer named '%s' not found in network. Skip it." % name)
            else:
                raise RuntimeError("Layer named '%s' not found." % name)
        else:
            g = f[name]
            layer = layer_index[name]
            if isinstance(layer, Layer):
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                for i, weight in enumerate(layer.weights):
                    weight_tuples.append((weight, np.asarray(g[weight_names[i]])))
            else:
                raise RuntimeError("Only layer can be saved into hdf5.")
    return weight_tuples


def _get_checkpoint_filename(filepath):
    from tensorflow.python.lib.io.file_io import is_directory
    from tensorflow.python.training.checkpoint_management import latest_checkpoint
    if is_directory(filepath):
        filepath = latest_checkpoint(filepath)
    if filepath is None:
        raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                         "given directory %s" % filepath)
    return filepath


def _load_checkpoint(filepath):
    from tensorflow.python.pywrap_tensorflow_internal import NewCheckpointReader
    filename = _get_checkpoint_filename(filepath)
    return NewCheckpointReader(filename)


def _normalize_name(name):
    if name.endswith(":0"):
        name = name[:-2]
    return name


def _load_weights_from_group(variables, reader, prefix=''):
    vars_to_shape = reader.get_variable_to_shape_map()
    _vars_to_restore = dict()
    skip_vars = []
    if prefix:
        index = variables[0].find(prefix)
        if index != 0:
            import warnings
            warnings.warn("Prefix %s starts from %d in var name which"
                          " isn't in the start of var name" % (prefix, index))
    for var in variables:
        var_name = var.name[:-2].replace(prefix, '', 1)
        var_shape = var.shape.as_list()

        skip_var = True

        if var_name in vars_to_shape and vars_to_shape[var_name] == var_shape:
            _vars_to_restore[var] = reader.get_tensor(var_name)
            skip_var = False
        if skip_var:
            skip_vars.append(var)

    return _vars_to_restore, skip_vars


def load_ckpt_weights(filepath, prefix=''):
    from tensorlib.engine.base_lib import get_session
    variables = fops.get_collection(fops.GraphKeys.GLOBAL_VARIABLES)
    reader = pywrap_tensorflow.NewCheckpointReader(filepath)
    vars_to_restore, skip_vars = _load_weights_from_group(
        variables, reader, prefix)
    assign_ops = []
    feed_dict = {}
    for var, value in vars_to_restore.items():
        if hasattr(var, '_assign_placeholder'):
            assign_placeholder = var._assign_placeholder
            assign_op = var._assign_op
        else:
            assign_placeholder = array_ops.placeholder(var.dtype, shape=value.shape)
            assign_op = var.assign(assign_placeholder)
            var._assign_placeholder = assign_placeholder
            var._assign_op = assign_op
        assign_ops.append(assign_op)
        feed_dict[assign_placeholder] = value
    get_session().run(assign_ops, feed_dict=feed_dict)
    logging.info("Restore weights from {}".format(filepath))
    logging.info("Values were loaded for {} tensors!".format(len(vars_to_restore.keys())))
    logging.info("Values were not loaded for {} tensors:".format(len(skip_vars)))
    for var in skip_vars:
        logging.info(" skip model vars:{}".format(var))
