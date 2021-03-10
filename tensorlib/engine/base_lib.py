from tensorflow.python import ops
from tensorflow.python.framework import ops as fops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.sparse import SparseTensor
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.client import device_lib
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2
import numpy as np

# {graph: learning_phase_placeholder(bool tensor)}
# global flag to control train or evaluate
_GRAPH_LEARNING_PHASES = {}

# This is the default internal TF session.
# It can be set manually via `set_session(sess)`.
_SESSION = None


def get_session(config=None,
                checkpoint_dir=None,
                checkpoint_path=None):
    global _SESSION
    if _SESSION is None:
        from tensorlib.training.sessions.core import BasicSessionCreator
        session_creator = BasicSessionCreator(
            config=config or config_pb2.ConfigProto(
                allow_soft_placement=True),
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path)
        _SESSION = session_creator.create_session()
        session = _SESSION
    else:
        if config is not None:
            import warnings
            warnings.warn("Session has already been created without specific"
                          " session_config: %s, you should invoke method"
                          " `get_session` manually in the beginning of your"
                          " program." % str(config))
        session = _SESSION
    if checkpoint_dir is None and checkpoint_path is None:
        with session.graph.as_default():
            all_vars = variables.global_variables()
            candidate_vars = []
            for var in all_vars:
                if not getattr(var, "_initialized", False):
                    candidate_vars.append(var)
            if candidate_vars:
                is_initialized = session.run([
                    variables.is_variable_initialized(v) for v in candidate_vars])
                uninitialized_vars = []
                for flag, v in zip(is_initialized, candidate_vars):
                    if not flag:
                        uninitialized_vars.append(v)
                    v._initialized = True
                if uninitialized_vars:
                    session.run(variables.variables_initializer(uninitialized_vars))
    return session


def clear_session():
    from tensorlib.engine.name_manager import reset_default_graph_uid
    global _SESSION
    global _GRAPH_LEARNING_PHASES
    fops.reset_default_graph()
    reset_default_graph_uid()
    _SESSION = None
    phase = array_ops.placeholder_with_default(
        False, shape=(), name='learning_phase')
    _GRAPH_LEARNING_PHASES = dict()
    _GRAPH_LEARNING_PHASES[fops.get_default_graph()] = phase


def learning_phase():
    graph = fops.get_default_graph()
    if graph not in _GRAPH_LEARNING_PHASES:
        phase = array_ops.placeholder_with_default(
            False, shape=(), name='learning_phase')
        _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]


def set_learning_phase(value):
    global _GRAPH_LEARNING_PHASES
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be 0 or 1')
    _GRAPH_LEARNING_PHASES[fops.get_default_graph()] = value


def batch_set_value(tuples):
    if not tuples:
        return
    assign_ops = []
    feed_dict = {}
    for x, value in tuples:
        value = np.asarray(value, dtype=dtype(x))
        tf_dtype = dtypes.as_dtype(x.dtype.name.split('_')[0])
        assign_placeholder = getattr(
            x, '_assign_placeholder', array_ops.placeholder(
                tf_dtype, shape=value.shape))
        assign_op = getattr(
            x, '_assign_op', x.assign(assign_placeholder))
        setattr(x, '_assign_placeholder', assign_placeholder)
        setattr(x, '_assign_op', assign_op)
        assign_ops.append(assign_op)
        feed_dict[assign_placeholder] = value
    get_session().run(assign_ops, feed_dict=feed_dict)


def is_gpu_available():
    local_device_protos = device_lib.list_local_devices()
    return any(x.device_type == 'GPU' for x in local_device_protos)


def is_sparse(x):
    return isinstance(x, SparseTensor)


def is_tensor(x):
    return isinstance(x, getattr(ops, '_TensorLike')) or ops.is_dense_tensor_like(x)


def is_variable(x):
    return isinstance(x, variables.Variable)


def is_placeholder(x):
    try:
        return x.op.type == 'Placeholder'
    except AttributeError:
        return False


def is_tensor_traceable(x):
    """
    Check whether tensor x is returned from subclass of base_layer.Layer or not
    which can be traced to where it come from (which layer, which node)
    Note: check detail in graph_node_ops
    """
    if not is_tensor(x):
        raise TypeError("Expecting an instance of 'Tensor',"
                        "but received an instance of {}".format(str(type(x))))
    return hasattr(x, '_history')


def assert_tensor_traceable(x):
    if not is_tensor_traceable(x):
        raise AttributeError("Missing attribute named `_history` (`graph_ops.History`"
                             " represents traceable information(node, idx)) in tensor %s"
                             " which indicates this tensor is node ?'s idx-th output"
                             " and use as bridge to build topological graph." % str(x))


def bool(tensor):
    return math_ops.cast(tensor, dtype=dtypes.bool)


def uint8(tensor):
    return math_ops.cast(tensor, dtype=dtypes.uint8)


def int8(tensor):
    return math_ops.cast(tensor, dtype=dtypes.int8)


def int16(tensor):
    return math_ops.cast(tensor, dtype=dtypes.int16)


def int32(tensor):
    return math_ops.cast(tensor, dtype=dtypes.int32)


def int64(tensor):
    return math_ops.cast(tensor, dtype=dtypes.int64)


def float32(tensor):
    return math_ops.cast(tensor, dtype=dtypes.float32)


def float32s(tensors):
    return [float32(tensor) for tensor in tensors]


def dtype(x):
    return x.dtype.base_dtype.name


def int_shape(tensor):
    try:
        shape = tensor.shape
        if not isinstance(shape, tuple):
            shape = tuple(shape.as_list())
        return shape
    except ValueError:
        return None


def get_shape(tensor):
    return tuple(
        i if i is not None else s
        for i, s in zip(
            int_shape(tensor), array_ops.unstack(
                array_ops.shape(tensor))))


def ndim(tensor):
    if is_tensor(tensor):
        return tensor.get_shape().ndims
    return None


def coord_indices(tensor, indices, axis=None):
    """
    Coordinate indices according to tensor, e.g.
    >> a = tf.random.normal((2, 3))
    tf.Tensor(
    [[-0.02075689  0.7327415  -0.70488846]
    [ 2.8714576  -0.2355751   0.33274603]], shape=(2, 3), dtype=float32)
    >> indices = tf.argmax(a, axis=0)
    tf.Tensor([1 0 1], shape=(3,), dtype=int64)  # can not use in gather method
    >> indices = coord_indices(a, indices, axis=0)
    tf.Tensor(
    [[1 0]
    [0 1]
    [1 2]], shape=(3, 2), dtype=int64)  # index along axis=0 was filled
    >> tf.gather_nd(a, indices)
    tf.Tensor([2.8714576  0.7327415  0.33274603], shape=(3,), dtype=float32)
    All we have to do is to calculate missing indices along dim: 0~axis, axis+1~rank
    which has same shape as indices, we just stack them together, we then get
    gather-style indices
    :param tensor: Tensor has rank at least is 1
    :param indices: indices(e.g. from tf.argmax) need to coordinate
        which must has rank that less than tensor's rank
    :param axis: indices along tensor's ? axis
    :return: coordinated indices
    """
    rank = ndim(tensor)
    if rank == 1:
        return indices
    shape = int_shape(tensor)
    ind_shape = int_shape(indices)
    if axis is None or axis == -1:
        axis = rank
    elif axis > rank:
        raise ValueError("Axis %d out of rank %d" % (axis, rank))
    if len(ind_shape) > rank:
        raise ValueError("Indices to coordinate must have rank"
                         " less than tensor's rank, but received:"
                         " %d vs %d" % (ind_shape[-1], rank))
    if axis != rank:
        shape = list(shape)
        shape[-1], shape[axis] = shape[axis], shape[-1]
    aux_indices = [math_ops.range(0, shape[i], dtype=indices.dtype)
                   for i in range(rank - 2, -1, -1)]
    aux_indices = array_ops.meshgrid(*aux_indices)
    aux_indices.reverse()
    if axis != rank:
        aux_indices.insert(axis, indices)
    else:
        aux_indices.append(indices)
    print(aux_indices)
    indices = array_ops.stack(aux_indices, axis=-1)
    return indices


def align_gather(tensor, indices, axis=None):
    """
    Gather elements from tensor according to indices
    if indices' rank = tensor's rank - 1, a one-hot
    mask calculated from indices along specific axis
    will be used to get required elements in tensor
    :param tensor: Tensor(vector or matrix format)
    :param indices: indices should has rank same to
     tensor or equal to tensor's rank - 1
    :param axis: only useful when indices' rank =
     tensor's rank - 1
    :return: values gather from tensor according to indices and axis
    """
    shape = int_shape(tensor)
    if len(shape) - ndim(indices) != 1:
        raise ValueError("Indices should have rank that equal to"
                         " tensor's rank - 1 or equal to tensor's"
                         " rank, but received: %d, %d" % (ndim(indices), len(shape)))
    if axis is None:
        axis = -1
    mask = array_ops.one_hot(indices, depth=shape[axis],
                             axis=axis, dtype=tensor.dtype)
    values = math_ops.reduce_sum(mask * tensor, axis=axis)
    return values


def amax(tensor, axis=None):
    """
    Return max values and corresponding indices of tensor along specific axis
    :param tensor: Tensor whose rank must >= 1
    :param axis: integer
    :return: max values, max indices
    """
    if not ndim(tensor):
        raise ValueError("Can not find max value and max indices in scalar")
    if not axis:
        axis = -1
    indices = math_ops.argmax(tensor, axis=axis)
    value = align_gather(tensor, indices=indices, axis=axis)
    return value, indices


def placeholder(shape=None,
                ndim=None,
                dtype=None,
                sparse=False,
                name=None):
    if dtype is None:
        dtype = dtypes.float32
    if not shape:
        if ndim:
            shape = (None,) * ndim
    if sparse:
        x = array_ops.sparse_placeholder(
            dtype, shape=shape, name=name)
    else:
        x = array_ops.placeholder(
            dtype, shape=shape, name=name)
    return x


def to_channels_first(x):
    x = tuple(x)
    assert len(x) >= 3
    return (x[0], x[-1]) + x[1: -1]


def to_channels_last(x):
    x = tuple(x)
    assert len(x) >= 3
    return (x[0],) + x[2:] + (x[1],)


def transpose_to_channels_first(x):
    perm = to_channels_first(range(len(int_shape(x))))
    return array_ops.transpose(x, perm=perm)


def transpose_to_channels_last(x):
    perm = to_channels_last(range(len(int_shape(x))))
    return array_ops.transpose(x, perm=perm)


def smart_cond(pred,
               true_fn=None,
               false_fn=None,
               name=None):
    if isinstance(pred, variables.Variable):
        return control_flow_ops.cond(
            pred, true_fn=true_fn, false_fn=false_fn, name=name)
    return smart_module.smart_cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)


def repeat_elements(x, repeat, axis):
    x_shape = int_shape(x)
    if x_shape[axis] is not None:
        splits = array_ops.split(value=x,
                                 num_or_size_splits=x_shape[axis],
                                 axis=axis)
        x_rep = [s for s in splits for _ in range(repeat)]
        return array_ops.concat(x_rep, axis=axis)
    # When x.shape[axis] is None, expand x along axis
    # repeat along axis + 1, then reshape to desired axis
    # Repeating
    aux_axis = axis + 1
    x_rep = array_ops.expand_dims(x, axis=aux_axis)
    multiples = np.ones(len(x_shape) + 1)
    multiples[aux_axis] = repeat
    x_rep = array_ops.tile(x_rep, multiples=multiples)
    # Merging
    multiples = np.delete(multiples, axis=aux_axis)
    multiples[axis] = repeat
    multiples = array_ops.constant(multiples, 'int32')
    multiples *= array_ops.shape(x)
    x_rep = array_ops.reshape(x_rep, shape=multiples)
    x_rep.set_shape(x_shape)  # axis set to None
    return x_rep


def resize_image(image,
                 height_factor,
                 width_factor,
                 data_format='NHWC',
                 interpolation='nearest'):
    if data_format[-1] == 'C':
        row, col = 1, 2
    else:
        row, col = 2, 3
    old_shape = int_shape(image)
    spatial = array_ops.shape(image)[row: col + 1]
    spatial *= array_ops.constant(
        np.array([height_factor, width_factor]), 'int32')
    if data_format[-1] != 'C':
        image = transpose_to_channels_last(image)
    if interpolation == 'nearest':
        image = image_ops.resize_nearest_neighbor(image, size=spatial)
    elif interpolation == 'bilinear':
        image = image_ops.resize_bilinear(image, size=spatial)
    else:
        raise ValueError("Unknown interpolation named " + interpolation)
    height = old_shape[row] or old_shape[row] * height_factor
    width = old_shape[col] or old_shape[col] * width_factor
    if data_format[-1] != 'C':
        image = transpose_to_channels_first(image)
        image.set_shape(old_shape[:2] + (height, width))
    else:
        image.set_shape((old_shape[0],) + (height, width) + (old_shape[-1],))
    return image


def resize_volumes(x,
                   factors,
                   data_format='NDHWC'):
    assert len(factors) == 3
    axes = (1, 2, 3) if data_format[-1] == 'C' else (2, 3, 4)
    for axis, factor in zip(axes, factors):
        x = repeat_elements(x, factor, axis)
    return x


def dot(x, y):
    x_dim = ndim(x)
    y_dim = ndim(y)
    if x_dim is not None and (x_dim > 2 or y_dim > 2):
        x_shape = get_shape(x)
        y_shape = get_shape(y)
        y_perm = list(range(y_dim))
        y_perm = [y_perm.pop(-2)] + y_perm
        xt = array_ops.reshape(x, [-1, x_shape[-1]])
        yt = array_ops.reshape(array_ops.transpose(
            y, perm=y_perm), [y_shape[-2], -1])
        return array_ops.reshape(math_ops.matmul(xt, yt),
                                 x_shape[:-1] + y_shape[:-2]
                                 + y_shape[-1:])
    if is_sparse(x):
        outputs = sparse_ops.sparse_tensor_dense_matmul(x, y)
    else:
        outputs = math_ops.mat_mul(x, y)
    return outputs


def batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = (axes, axes)
    x_dim = ndim(x)
    y_dim = ndim(y)
    if axes is None:
        axes = [x_dim - 1, y_dim - 2]
    diff = x_dim - y_dim
    if diff > 0:
        y = array_ops.reshape(
            y, array_ops.concat([
                array_ops.shape(y), [1] * diff], axis=0),)
    elif diff < 0:
        x = array_ops.reshape(
            x, array_ops.concat([
                array_ops.shape(x), [1] * -diff], axis=0))
    if x_dim == 2 and y_dim == 2:
        if axes[0] == axes[1]:
            outputs = math_ops.reduce_sum(
                math_ops.multiply(x, y), axes[0])
        else:
            outputs = math_ops.reduce_sum(
                math_ops.multiply(array_ops.transpose(
                    x, [1, 0], y), axes[1]))
    else:
        adj_x = None if axes[0] == x_dim - 1 else True
        adj_y = True if axes[1] == y_dim - 1 else None
        outputs = math_ops.matmul(
            x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_dim > y_dim:
            idx = x_dim + y_dim - 3
        else:
            idx = x_dim - 1
        outputs = array_ops.squeeze(outputs, list(range(idx, idx + diff)))
    if ndim(outputs) == 1:
        outputs = array_ops.expand_dims(outputs, axis=1)
    return outputs


def bias_add(value, bias, data_format='NHWC'):
    dim = ndim(value)
    bias_shape = int_shape(bias)
    if len(bias_shape) != 1 and len(bias_shape) != dim - 1:
        raise ValueError("Expect bias's dim is 1 or %d, but received %d" % (
            dim - 1, len(bias_shape)))
    channel_last = data_format[-1] == 'C'
    if dim > 1:
        if channel_last:
            if len(bias_shape) == 1:
                if dim == 4:
                    return nn.bias_add(value, bias,
                                       data_format=data_format)
                shape = (1,) * (dim - 1) + (bias_shape[0],)
            else:
                shape = (1,) + bias_shape
        else:
            if len(bias_shape) == 1:
                # 'NCHW' is not available on cpu
                if dim == 4 and is_gpu_available():
                    return nn.bias_add(value, bias,
                                       data_format=data_format)
                shape = (1, bias_shape[0]) + (1,) * (dim - 2)
            else:
                shape = (1, bias_shape[-1]) + bias_shape[:-1]
        value += array_ops.reshape(bias, shape)
    else:
        value = nn.bias_add(value=value, bias=bias)
    return value


def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='VALID',
                     data_format='NHWC',
                     dilation_rate=(1, 1)):
    strides = (1,) + strides + (1,) if data_format[-1] == 'C'\
        else (1, 1) + strides
    # deconv's output_shape can not contain "None"
    # if had "None", replace it with -1
    if output_shape[0] is None:
        batch = int_shape(x)[0]
        output_shape = (batch if batch else -1,) + tuple(output_shape[1:])
    if dilation_rate == (1, 1):
        x = nn.conv2d_transpose(value=x,
                                filter=kernel,
                                output_shape=output_shape,
                                strides=strides,
                                padding=padding,
                                data_format=data_format)
    else:
        assert dilation_rate[0] == dilation_rate[1]
        # atrous only support NHWC
        if data_format[-1] != 'C':
            x = transpose_to_channels_last(x)
            output_shape = to_channels_last(output_shape)
        x = nn.atrous_conv2d_transpose(value=x,
                                       filters=kernel,
                                       output_shape=output_shape,
                                       rate=dilation_rate[0],
                                       padding=padding)
        if data_format[-1] != 'C':
            x = transpose_to_channels_first(x)
    return x


def separable_conv1d(value,
                     depthwise_kernel,
                     pointwise_kernel,
                     bias=None,
                     strides=(1,),
                     padding='VALID',
                     data_format='NWC',
                     dilation_rate=(1,),
                     name=None):
    if data_format[-1] == 'C':
        strides = (1,) + strides * 2 + (1,)
        data_format = 'NHWC'
        axis = 1
    else:
        strides = (1, 1) + strides * 2
        data_format = 'NCHW'
        axis = 2
    value = array_ops.expand_dims(value, axis=axis)
    depthwise_kernel = array_ops.expand_dims(depthwise_kernel, 0)
    pointwise_kernel = array_ops.expand_dims(pointwise_kernel, 0)
    value = nn.separable_conv2d(
        input=value,
        depthwise_filter=depthwise_kernel,
        pointwise_filter=pointwise_kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilation_rate,
        name=name)
    if bias is not None:
        value = nn.bias_add(value, bias, data_format=data_format)
    value = array_ops.squeeze(value, axis=axis)
    return value


def _to_channel_first_bias(b):
    channel_size = int(b.shape[1])
    new_shape = (channel_size, 1, 1)
    return array_ops.reshape(b, new_shape)


def batch_normalization(batch_inputs,
                        mean,
                        variance,
                        offset,
                        scale,
                        epsilon,
                        data_format,
                        name=None):
    """
    param batch_inputs: shape of [num_components, batch, 1] or [channels, batch, h, w]
    :return shape of [batch, num_components, 1] or [batch, h, w, channel] or [batch, channel, h, w]
    """
    with ops.name_scope(name, 'batchnorm',
                        [batch_inputs, mean,
                         variance, scale, offset]):
        shape = [1] * (len(batch_inputs.shape) - 1) + [-1] if data_format[-1] == 'C' \
            else [1, -1] + [1] * (len(batch_inputs.shape) - 2)
        mean = array_ops.reshape(mean, shape)
        variance = array_ops.reshape(variance, shape)
        offset = array_ops.reshape(offset, shape)
        scale = array_ops.reshape(scale, shape)

        inv = math_ops.rsqrt(variance + epsilon)
        if scale is not None:
            inv *= scale
        a = math_ops.cast(inv, batch_inputs.dtype)
        b = math_ops.cast(offset - mean * inv if offset is not None
                          else -mean * inv, batch_inputs.dtype)
        if data_format[-1] != 'C':
            a = _to_channel_first_bias(a)
            b = _to_channel_first_bias(b)
        outputs = math_ops.add(math_ops.multiply(batch_inputs, a), b)
    return outputs


def rnn(step_fn,
        inputs,
        initial_states,
        go_backwards=False,
        unroll=False,
        input_length=None,
        name='rnn_block'):
    with ops.name_scope(name):
        dim = ndim(inputs)
        if dim < 3:
            raise ValueError("Input should be at least 3D")
        perm = [1, 0] + list(range(2, dim))
        inputs = array_ops.transpose(inputs, perm=perm, name='to_time_major')
        if unroll:
            assert int_shape(inputs)[0] is not None,\
                "Unrolling requires a fixed number of time steps"
            states = initial_states
            successive_states = []
            successive_outputs = []
            input_list = array_ops.unstack(inputs)
            if go_backwards:
                input_list.reverse()
            for x in input_list:
                outputs, states = step_fn(x, states)
                successive_outputs.append(outputs)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = array_ops.stack(successive_outputs)
        else:
            if go_backwards:
                inputs = array_ops.reverse(inputs, axis=0)
            states = tuple(initial_states)
            time_steps = array_ops.shape(inputs)[0]
            outputs, _ = step_fn(inputs[0], initial_states)
            output_ta = tensor_array_ops.TensorArray(
                dtype=outputs.dtype,
                size=time_steps,
                tensor_array_name='output_ta')
            input_ta = tensor_array_ops.TensorArray(
                dtype=inputs.dtype,
                size=time_steps,
                tensor_array_name='input_ta')
            # unstack inputs and write into input array
            input_ta = input_ta.unstack(inputs)
            time = array_ops.constant(0, dtype='int32', name='time')

            def _step(_time, _output_ta, *_states):
                current_input = input_ta.read(_time)
                output, _new_states = step_fn(current_input, tuple(_states))
                for state, new_state in zip(_states, _new_states):
                    new_state.set_shape(state.get_shape())
                _output_ta = _output_ta.write(_time, output)
                return (_time + 1, _output_ta) + tuple(_new_states)
            final_outputs = control_flow_ops.while_loop(
                cond=lambda _time, *_: _time < time_steps,
                body=_step,
                loop_vars=(time, output_ta) + states,
                parallel_iterations=32,
                swap_memory=True,
                maximum_iterations=input_length)
            last_time = final_outputs[0]
            output_ta = final_outputs[1]
            new_states = final_outputs[2:]
            outputs = output_ta.stack()
            last_output = output_ta.read(last_time - 1)
        perm = [1, 0] + list(range(2, ndim(outputs)))
        outputs = array_ops.transpose(outputs, perm=perm)
    return last_output, outputs, new_states


def affine_grid(theta, size: (list, tuple), name='affine_grid'):
    with ops.name_scope(name):
        x = gen_math_ops.lin_space(-1., 1., size[1])
        y = gen_math_ops.lin_space(-1., 1., size[2])
        x_t, y_t = array_ops.meshgrid(x, y)
        x_t = array_ops.reshape(x_t, shape=(-1,))
        y_t = array_ops.reshape(y_t, shape=(-1,))
        ones = array_ops.ones_like(x_t)
        grids = array_ops.stack([x_t, y_t, ones])
        grids = array_ops.expand_dims(grids, axis=0)
        grids = array_ops.tile(grids, multiples=array_ops.stack([size[0], 1, 1]))
        grids = float32(grids)
        theta = float32(theta)
        grids = math_ops.matmul(theta, grids)
        grids = array_ops.reshape(grids, shape=(size[0], 2, size[1], size[2]))
    return grids


def grid_sample(inputs, grid, padding_mode='CONSTANT', name='grid_sample'):
    def _get_pixel(image, _y, _x):
        b, _h, _w = image.get_shape().as_list()[0: -1]
        batch_idx = array_ops.reshape(math_ops.range(b), shape=(b, 1, 1))
        batch_idx = array_ops.tile(batch_idx, multiples=(1, _h - 1, _w - 1))
        indices = array_ops.stack([batch_idx, _y, _x], axis=3)
        return array_ops.gather_nd(image, indices)
    with ops.name_scope(name):
        x_s = grid[:, 0, :, :]
        y_s = grid[:, 1, :, :]
        h, w = inputs.get_shape().as_list()[1: -1]
        images = array_ops.pad(inputs, array_ops.constant(
            ((0, 0), (0, 1), (0, 1), (0, 0))), mode=padding_mode)
        h = int32(h)
        w = int32(w)
        zero = array_ops.zeros([], dtypes.int32)
        x = (math_ops.multiply(x_s + 1., float32(w))) * 0.5
        y = (math_ops.multiply(y_s + 1., float32(h))) * 0.5
        x0 = clip_ops.clip_by_value(int32(math_ops.floor(x)), zero, w)
        x1 = clip_ops.clip_by_value(x0 + 1, zero, w)
        y0 = clip_ops.clip_by_value(int32(math_ops.floor(y)), zero, h)
        y1 = clip_ops.clip_by_value(y0 + 1, zero, h)
        ptl = _get_pixel(images, y0, x0)
        pbl = _get_pixel(images, y1, x0)
        ptr = _get_pixel(images, y0, x1)
        pbr = _get_pixel(images, y1, x1)
        x0 = float32(x0)
        x1 = float32(x1)
        y0 = float32(y0)
        y1 = float32(y1)
        wtl = array_ops.expand_dims(math_ops.multiply(
            math_ops.subtract(x1, x), math_ops.subtract(y1, y)), axis=3)
        wbl = array_ops.expand_dims(math_ops.multiply(
            math_ops.subtract(x1, x), math_ops.subtract(y, y0)), axis=3)
        wtr = array_ops.expand_dims(math_ops.multiply(
            math_ops.subtract(x, x0), math_ops.subtract(y1, y)), axis=3)
        wbr = array_ops.expand_dims(math_ops.multiply(
            math_ops.subtract(x, x0), math_ops.subtract(y, y0)), axis=3)
        outputs = math_ops.add_n(
            [math_ops.multiply(wtl, ptl), math_ops.multiply(wbl, pbl),
             math_ops.multiply(wtr, ptr), math_ops.multiply(wbr, pbr)])
    return outputs
