import h5py
from tensorlib.engine.base_layer import Layer
from tensorlib.engine.base_lib import batch_set_value, get_session
from tensorlib.utils import to_list
from distutils.version import LooseVersion
from inspect import isgeneratorfunction
from tensorflow import logging
from tensorflow import train


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


def save_hdf5_weights(file_path, lib_model):
    with h5py.File(file_path, "w")as f:
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


def load_hdf5_weights(file_path,
                      lib_model,
                      allow_skip=False):
    import warnings
    f = h5py.File(file_path, 'r')
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


def _get_checkpoint_filename(file_path):
    from tensorflow.python.lib.io.file_io import is_directory
    from tensorflow.python.training.checkpoint_management import latest_checkpoint
    if is_directory(file_path):
        file_path = latest_checkpoint(file_path)
    if file_path is None:
        raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                         "given directory %s" % file_path)
    return file_path


def _load_checkpoint(file_path):
    from tensorflow.python.pywrap_tensorflow_internal import NewCheckpointReader
    filename = _get_checkpoint_filename(file_path)
    return NewCheckpointReader(filename)


def _normalize_name(name):
    if name.endswith(":0"):
        name = name[:-2]
    return name


def _get_vars_to_restore(model_vars, prefixes, vars_to_shape):
    restore_vars = dict()
    skip_vars = []
    state = dict()
    for prefix in prefixes:
        state[prefix] = 0
    for var in model_vars:
        name = var.name[:-2]
        shape = var.shape.as_list()
        skip_var = True
        for prefix in prefixes:
            name = name.replace(prefix, '')
            if name in vars_to_shape and vars_to_shape[name] == shape:
                restore_vars[name] = var
                state[prefix] += 1
                skip_var = False
                break
        if skip_var:
            skip_vars.append(var)

    return restore_vars, skip_vars, state


def load_checkpoint_weights(file_path, lib_model, prefixes=None):
    if prefixes is None:
        prefixes = ''
    prefixes = to_list(prefixes)
    reader = _load_checkpoint(file_path)
    weights = list(lib_model.weights)
    vars_to_shape = reader.get_variable_to_shape_map()
    restored_vars, skipped_vars, stated = _get_vars_to_restore(
        weights, prefixes, vars_to_shape)
    for prefix, num in stated.items():
        logging.info("For the prefix '{0}' were found {1} weights".format(prefix, num))
    try:
        train.init_from_checkpoint(file_path, restored_vars)
        logging.info("Values were loaded for {} tensors!".format(len(restored_vars.keys())))
        logging.info("Values were not loaded for {} tensors:".format(len(skipped_vars)))
        for var in skipped_vars:
            logging.info("skip values {}".format(var))
    except ValueError as exception:
        logging.error("Weights was not loaded at all!")
        logging.error(exception)
        exit(1)


# def load_checkpoint_weights(file_path, lib_model, global_step=None):
#     import warnings
#     reader = _load_checkpoint(file_path)
#     if global_step is not None:
#         value = reader.get_tensor(_normalize_name(global_step.op.name))
#         global_step.assign(value)
#     for var in lib_model.weights:
#         if reader.has_tensor(_normalize_name(var.op.name)):
#             var_value = reader.get_tensor(_normalize_name(var.op.name))
#             var.assign(var_value)
#         else:
#             warnings.warn('Variable %s missing in checkpoint %s', var, file_path)
