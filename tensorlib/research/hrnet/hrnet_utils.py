from tensorlib.research.resnet_v1_beta.resnet_v1_beta import ResNet_V1_beta_block, ResNet_V1_small_beta_block
import tensorlib as lib
import tensorflow as tf
from collections import namedtuple


BLOCKS = {'bottleneck': ResNet_V1_beta_block,
          'lite_bottleneck': ResNet_V1_small_beta_block}


class HRStage(namedtuple('HRStage', ['scope', 'module_fn', 'args'])):
    pass


def stack_stages(net, stages):
    nets = [net]
    for i, stage in enumerate(stages):
        with tf.name_scope(stage.scope, 'stage', nets):
            for j, module in enumerate(stage.args):
                point_name = 'module_%d' % (j + 1)
                nets = stage.module_fn(nets, **dict(module, name=point_name))
            if i + 1 < len(stages):
                block_type = stages[i + 1].args[0]['block_type']
                expansion = 4 if block_type == 'bottleneck' else 1
                depths = [depth * expansion for depth in stages[i + 1].args[0]['depths']]
                nets = fusion_block(nets, depths=depths)
    return nets


def fusion_block(nets, depths, fuse_method='sum', name='fusion'):
    """
    Exchange information between different spatial feature maps
    Implementation better show in graph: for e.g. (3 nets, 4 depths)
    prepare net vector [None, None, None]
    i is iterator of nets in [0, 1, 2], j is iterator of depths in [0, 1, 2, 3]
    set nets = [x, y, z], ↑: up sample, ↓: down sample
    iteration procedure:
    iter 0: net vector = [x, y↑, z↑] -> do fusion
    iter 1: net vector = [x↓, y, z↑] -> do fusion
    iter 2: net vector = [x↓↓, y↓, z] -> do fusion
    iter 3: net vector = [x↓↓↓, y↓↓, z↓] -> do fusion
    we can find an rule: original tensors in diagonal
    down sample tensors below diagonal, and k+1 iter's tensors only need
    down sample one time using k iter's tensors
    up sample tensors upon diagonal
    In this way, we can reuse previous iteration's down sample results
    and only in O(n^2) cost
    """
    assert len(depths) - len(nets) == 1, \
        "num of depths({:d}) - num of nets({:d}) must be 1".format(len(depths), len(nets))
    with tf.name_scope(name, values=nets):
        spatial = [lib.engine.int_shape(net)[1:-1] for net in nets]
        net_vector = [None] * len(nets)
        outputs = []
        # Due to python's lazy evaluation mechanism,
        # we can not pass a spatial[i] with a variable i into lambda.
        # lambda only remembers the last value of i, instead of the value
        # when it was passed to lambda, so we need to make it to be a local
        # variable of lambda, it can be done in two ways
        # 1. currying, exactly the way we do in the below codes
        # 2. use 'lambda x, size=spatial[i]: ....'
        resize_fn = lambda size: lambda x: tf.image.resize_bilinear(
            x, size=size, align_corners=True)
        for i, depth in enumerate(depths):
            for j in range(len(nets)):
                num = i - j
                if num == 0:
                    net_vector[j] = nets[j]
                elif num < 0:
                    net_vector[j] = lib.layers.Lambda(
                        resize_fn(spatial[i]),
                        name='upsample')(nets[j])
                    net_vector[j] = lib.contrib.Conv2D(
                        out_channels=depth, kernel_size=1,
                        name='c_reduce')(net_vector[j])
                else:
                    net_vector[j] = lib.contrib.Conv2D(
                        out_channels=depth, kernel_size=3,
                        stride=2, name='downsample')(net_vector[j])
            if len(net_vector) > 1:
                if fuse_method == 'sum':
                    outputs.append(lib.layers.Lambda(lambda x: tf.add_n(x), name='sum_fusion')(*net_vector))
                else:
                    outputs.append(lib.layers.concat(*net_vector, axis=-1))
            else:
                outputs.append(net_vector[0])
    return outputs


def hrnet_arg_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        activation_fn='relu',
        use_batch_norm=True,
        use_weight_standardization=False):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    with lib.engine.arg_scope(
            [lib.contrib.WSConv2D],
            kernel_regularizer=lib.regularizers.l2(weight_decay),
            kernel_initializer='truncated_normal',
            activation_fn=activation_fn,
            normalizer_fn=lib.layers.BatchNorm if use_batch_norm else None,
            use_weight_standardization=use_weight_standardization):
        with lib.engine.arg_scope([lib.layers.BatchNorm], **batch_norm_params) as arg_sc:
            return arg_sc
