from tensorlib.utils.generic_util import valid_value, to_tuple


__all__ = ["conv_output_length", "normalize_data_format",
           "normalize_padding", "normalize_tuple",
           "deconv_output_length"]


_DATA_FORMAT = (('NWC', 'NCW'), ('NHWC', 'NCHW'), ('NDHWC', 'NCDHW'))


def conv_output_length(
        input_length,
        kernel_size,
        padding,
        stride,
        dilation=1):
    if input_length is None:
        return None
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    if padding == 'SAME':
        output_length = input_length
    else:
        output_length = input_length - dilated_kernel_size + 1
    return (output_length + stride - 1) // stride


def deconv_output_length(
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation=1):
    if input_length is None:
        return None
    dilated_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    if output_padding:
        if padding == 'VALID':
            pad = dilated_kernel_size // 2
        else:
            pad = 0
        output_length = (input_length - 1) * stride + dilated_kernel_size - 2 * pad + output_padding
    else:
        if padding == 'VALID':
            output_length = input_length * stride + max(dilated_kernel_size - stride, 0)
        else:
            output_length = input_length * stride
    return output_length


def normalize_tuple(value, n: int, name: str):
    """
    :param value: scalar or tuple or list
    :param n: 1d->1, 2d->2, 3d->3
    :param name: name of value
    # Usage (2d)
    >> kernel_size = 3
    >> normalize_tuple(kernel_size, 2, 'kernel_size', 'NHWC')
    >> [3, 3]
    """
    if isinstance(value, int):
        return (value, ) * n
    elif hasattr(value, '__iter__'):
        if len(value) != n:
            raise ValueError("Except {}'s length of {:d}, but received {:d}".format(name, n, len(value)))
        if any(not isinstance(v, int) for v in value):
            raise ValueError("All value in {} must be integer, but received {}".format(name, str(value)))
    return to_tuple(value)


def normalize_data_format(value, n: int):
    """
    :param value: str
    :param n: 1d->1, 2d->2, 3d->3
    # Usage (2d)
    >> data_format = 'channels_last'
    >> normalize_data_format(data_format, 2)
    >> 'NHWC'
    """
    if value is None:
        return _DATA_FORMAT[n-1][0]
    elif value == 'channels_first':
        return _DATA_FORMAT[n-1][1]
    elif value == 'channels_last':
        return _DATA_FORMAT[n-1][0]
    data_format = value.upper()
    data_format = valid_value(data_format, _DATA_FORMAT[n-1])
    return data_format


def normalize_padding(value):
    """
    :param value: str
    # Usage
    >> padding = 'same'
    >> normalize_padding(padding)
    >> 'SAME'
    """
    if isinstance(value, (list, tuple)):
        return value
    padding = valid_value(value.upper(), ('VALID', 'SAME'))
    return padding
