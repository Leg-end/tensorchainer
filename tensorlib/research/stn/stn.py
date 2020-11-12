import tensorlib as lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class STN(lib.Network):
    def __init__(self,
                 num_theta,
                 out_size=None,
                 pad_mode='CONSTANT',
                 data_format='channels_first',
                 **kwargs):
        super(STN, self).__init__(**kwargs)
        if out_size:
            assert len(out_size) == 2 and all(isinstance(x, int) for x in out_size)
            out_size = tuple(out_size)
        self.pad_mode = pad_mode
        self.out_size = out_size
        self.data_format = lib.utils.normalize_data_format(data_format, 2)
        self.num_theta = num_theta
        self.localisation = lib.layers.Dense(num_units=num_theta, name='localisation')

    def transform_theta(self, theta):
        zero = array_ops.zeros((lib.engine.int_shape(theta)[0], 1))
        if self.num_theta == 4:  # attention matrix
            # [a, b, c, d] => [[a, 0, b], [0, c, d]]
            theta = array_ops.stack([math_ops.sigmoid(theta[:, 0:1]), zero,
                                     math_ops.tanh(theta[:, 1:2]), zero,
                                     math_ops.sigmoid(theta[:, 2:3]),
                                     math_ops.tanh(theta[:, 3:4])], axis=1)
        ret = array_ops.reshape(theta, (2, 3))
        return ret

    def forward(self, inputs):
        out_size = lib.engine.int_shape(inputs)
        if self.out_size:
            out_size = out_size[0:1] + self.out_size
        else:
            out_size = out_size[:-1] if self.data_format[-1] == 'C' else out_size[2:]
        theta = self.localisation(inputs)
        theta = self.transform_theta(theta)
        grids = lib.engine.affine_grid(theta, out_size)
        outputs = lib.engine.grid_sample(inputs, grids, padding_mode=self.pad_mode)
        return outputs
