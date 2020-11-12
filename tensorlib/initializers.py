from tensorflow.python.ops.init_ops import Constant
from tensorflow.python.ops.init_ops import GlorotNormal
from tensorflow.python.ops.init_ops import GlorotUniform
from tensorflow.python.ops.init_ops import he_normal
from tensorflow.python.ops.init_ops import he_uniform
from tensorflow.python.ops.init_ops import Identity
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops.init_ops import lecun_normal
from tensorflow.python.ops.init_ops import lecun_uniform
from tensorflow.python.ops.init_ops import Ones
from tensorflow.python.ops.init_ops import Orthogonal
from tensorflow.python.ops.init_ops import RandomNormal as TFRandomNormal
from tensorflow.python.ops.init_ops import RandomUniform as TFRandomUniform
from tensorflow.python.ops.init_ops import TruncatedNormal as TFTruncatedNormal
from tensorflow.python.ops.init_ops import VarianceScaling
from tensorflow.python.ops.init_ops import Zeros
# from tensorflow.python.ops.init_ops_v2 import Constant as ConstantV2
# from tensorflow.python.ops.init_ops_v2 import GlorotNormal as GlorotNormalV2
# from tensorflow.python.ops.init_ops_v2 import GlorotUniform as GlorotUniformV2
# from tensorflow.python.ops.init_ops_v2 import he_normal as he_normalV2
# from tensorflow.python.ops.init_ops_v2 import he_uniform as he_uniformV2
# from tensorflow.python.ops.init_ops_v2 import Identity as IdentityV2
# from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2
# from tensorflow.python.ops.init_ops_v2 import lecun_normal as lecun_normalV2
# from tensorflow.python.ops.init_ops_v2 import lecun_uniform as lecun_uniformV2
# from tensorflow.python.ops.init_ops_v2 import Ones as OnesV2
# from tensorflow.python.ops.init_ops_v2 import Orthogonal as OrthogonalV2
# from tensorflow.python.ops.init_ops_v2 import RandomNormal as RandomNormalV2
# from tensorflow.python.ops.init_ops_v2 import RandomUniform as RandomUniformV2
# from tensorflow.python.ops.init_ops_v2 import TruncatedNormal as TruncatedNormalV2
# from tensorflow.python.ops.init_ops_v2 import VarianceScaling as VarianceScalingV2
# from tensorflow.python.ops.init_ops_v2 import Zeros as ZerosV2
from tensorflow.python.framework import dtypes
from tensorlib import saving
import numpy as np
import math
from tensorflow.python.ops import array_ops


class TruncatedNormal(TFTruncatedNormal):

    def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
        super(TruncatedNormal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class RandomUniform(TFRandomUniform):
    def __init__(self, minval=-0.05, maxval=0.05, seed=None,
                 dtype=dtypes.float32):
        super(RandomUniform, self).__init__(
            minval=minval, maxval=maxval, seed=seed, dtype=dtype)


class RandomNormal(TFRandomNormal):
    def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
        super(RandomNormal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class BiLinear(Initializer):

    def __call__(self, shape, dtype=dtypes.float32, partition_info=None):
        value = np.zeros(shape)
        f = math.ceil(shape[1] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        ys, xs = np.meshgrid(np.linspace(0, shape[0] - 1, shape[0]),
                             np.linspace(0, shape[1] - 1, shape[1]))
        bilinear = (1 - np.abs(ys / f - c)) * (1 - np.abs(xs / f - c))
        for i in range(shape[3]):
            value[:, :, i, i] = bilinear
        return array_ops.constant(value, dtype=dtype, shape=shape)


def get(identifier):
    if identifier is None or callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if identifier in globals():
            return globals()[identifier]()
        else:
            raise ValueError("Can not find initializer named `{}`"
                             " in module `initializers`".format(identifier))
    elif isinstance(identifier, dict):
        if 'class_name' not in identifier:
            raise ValueError("Identifier is illegal, ", str(identifier))
        return saving.load_dict(identifier)
    else:
        raise ValueError("Could not interpret identifier:", identifier)


zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform
variance_scaling = VarianceScaling

