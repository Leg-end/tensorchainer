import tensorlib as lib
import tensorflow as tf
from tensorlib.research.resnet_v1_beta import resnet_v1_utils

_DEFAULT_MULTI_GRID = [1, 1, 1]
_DEFAULT_MULTI_GRID_RESNET_18 = [1, 1]


class BasicBlock(lib.engine.Network):
    @lib.engine.add_arg_scope
    def __init__(self,
                 depth_in,
                 depth,
                 stride,
                 unit_rate=1,
                 rate=1,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        if depth == depth_in:
            self.shortcut = resnet_v1_utils.sub_sample(
                stride, 'shortcut')
        else:
            self.shortcut = lib.contrib.WSConv2D(
                out_channels=depth,
                kernel_size=1,
                strides=stride,
                activation=None,
                name='shortcut')
        self.conv1 = resnet_v1_utils.conv2d_same(
            out_channels=depth,
            kernel_size=3,
            stride=1,
            rate=rate * unit_rate,
            name='conv1')
        with lib.engine.arg_scope([lib.contrib.WSConv2D], activation=None):
            self.conv2 = resnet_v1_utils.conv2d_same(
                out_channels=depth,
                kernel_size=3,
                stride=stride,
                rate=rate * unit_rate,
                name='conv2')
        self.add = lib.layers.ElementWise('add', activation='relu')

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        residual = self.conv1(inputs)
        residual = self.conv2(residual)
        outputs = self.add(shortcut, residual)
        return outputs


class Bottleneck(lib.engine.Network):
    @lib.engine.add_arg_scope
    def __init__(self,
                 depth_in,
                 depth,
                 depth_bottleneck,
                 stride,
                 unit_rate=1,
                 rate=1,
                 **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        if depth == depth_in:
            self.shortcut = resnet_v1_utils.sub_sample(
                stride, 'shortcut')
        else:
            self.shortcut = lib.contrib.WSConv2D(
                out_channels=depth,
                kernel_size=1,
                strides=stride,
                activation=None,
                name='shortcut')
        self.conv1 = lib.contrib.WSConv2D(
            out_channels=depth_bottleneck,
            kernel_size=1,
            strides=1,
            name='conv1')
        self.conv2 = resnet_v1_utils.conv2d_same(
            out_channels=depth_bottleneck,
            kernel_size=3,
            stride=stride,
            rate=rate * unit_rate,
            name='conv2')
        self.conv3 = lib.contrib.WSConv2D(
            out_channels=depth,
            kernel_size=1,
            strides=1,
            activation=None,
            name='conv3')
        self.add = lib.layers.ElementWise('add', activation='relu')

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        residual = self.conv1(inputs)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        outputs = self.add(shortcut, residual)
        return outputs


def ResNet_V1_beta_block(scope, base_depth, num_units, stride):
    return resnet_v1_utils.Block(scope, Bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
        'unit_rate': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
        'unit_rate': 1
    }])


def ResNet_V1_small_beta_block(scope, base_depth, num_units, stride):
    block_args = []
    for _ in range(num_units - 1):
        block_args.append({'depth': base_depth, 'stride': 1, 'unit_rate': 1})
    block_args.append({'depth': base_depth, 'stride': stride, 'unit_rate': 1})
    return resnet_v1_utils.Block(scope, BasicBlock, block_args)


class ResNetV1Beta(lib.engine.Network):
    def __init__(self,
                 blocks,
                 num_classes=None,
                 is_training=None,
                 base_only=False,
                 global_pool=True,
                 extract_blocks=None,
                 output_stride=None,
                 root_block=None,
                 **kwargs):
        super(ResNetV1Beta, self).__init__(**kwargs)
        if extract_blocks is not None:
            extract_blocks = lib.utils.to_list(extract_blocks)
            self.extract_hook = lib.hooks.ExtractHook(
                out_names=extract_blocks, prefix=self.name)
        else:
            self.extract_hook = lib.hooks.EmptyHook()
        if not base_only and output_stride not in [32, None]:
            raise ValueError("As the `base_only` is set to `False`, `output_stride` can only be 32 or None, "
                             "but given %d." % output_stride)
        if output_stride is not None and output_stride not in [8, 16, 32]:
            raise ValueError('Only allowed output_stride values are 8, 16, 32.')
        self.base_only = base_only
        self.global_pool = global_pool
        self.bottlenecks = lib.Sequential(name='')
        self.num_classes = num_classes
        if is_training is not None:
            arg_scope = lib.engine.arg_scope(
                [lib.layers.BatchNorm], trainable=is_training)
        else:
            arg_scope = lib.engine.arg_scope([])
        with self.extract_hook:
            with arg_scope:
                if root_block is None:
                    self.conv1 = resnet_v1_utils.conv2d_same(
                        out_channels=64, kernel_size=7,
                        stride=2, name='conv1')
                else:
                    self.conv1 = root_block
                self.pool1 = lib.layers.MaxPool2D(
                    kernel_size=3, strides=2,
                    padding='SAME', name='pool1')
                self.stack_blocks_dense(blocks, 64, output_stride)
                if not base_only:
                    if global_pool:
                        self.gpool = lib.layers.GlobalAvgPool(name='pool5')
                    if num_classes is not None:
                        self.logits = lib.contrib.WSConv2D(
                            out_channels=num_classes,
                            kernel_size=1,
                            activation=None,
                            normalizer=None,
                            use_weight_standardization=False,
                            name='logits')
                        self.sp_squeeze = lib.layers.Lambda(
                            lambda x: tf.squeeze(x, axis=[1, 2]),
                            name='spatial_squeeze')

    def stack_blocks_dense(self, blocks, in_channels, output_stride=None):
        current_stride = 4
        rate = 1
        for block in blocks:
            block_layer = lib.Sequential(name=block.name)
            block_stride = 1
            for i, unit in enumerate(block.args):
                unit['name'] = 'unit_%d' % (i + 1)
                unit['depth_in'] = in_channels
                if output_stride is not None and current_stride == output_stride:
                    block_layer.append(block.unit(**dict(unit, stride=1)))
                    rate *= unit.get('stride', 1)
                else:
                    block_layer.append(block.unit(**unit))
                    current_stride *= unit.get('stride', 1)
                    if output_stride is not None and current_stride > output_stride:
                        raise ValueError("The target output_stride can not be reached.")
                in_channels = unit.get('depth')
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                block_layer.append(resnet_v1_utils.sub_sample(block_stride))
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError("The target output_stride can not be reached.")
            self.bottlenecks.append(block_layer)
        if output_stride is not None and current_stride != output_stride:
            raise ValueError("The target output_stride can not be reached.")

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.pool1(outputs)
        outputs = self.bottlenecks(outputs)
        if isinstance(self.extract_hook, lib.hooks.ExtractHook):
            return self.extract_hook.get_extract()[1]
        if not self.base_only:
            if self.global_pool:
                outputs = self.gpool(outputs)
            if self.num_classes is not None:
                outputs = self.logits(outputs)
                outputs = self.sp_squeeze(outputs)
        return outputs


def ResNetV1_18(num_classes=None,
                is_training=None,
                global_pool=False,
                base_only=False,
                min_base_depth=8,
                depth_multiplier=1,
                output_stride=None,
                extract_blocks=None,
                multi_grid=None,
                name='resnet_v1_18'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID_RESNET_18
    elif len(multi_grid) != 2:
        raise ValueError("Expect multi_grid to have length 2.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    block4_args = []
    for rate in multi_grid:
        block4_args.append({'depth': depth_func(512), 'stride': 1, 'unit_rate': rate})

    blocks = [
        ResNet_V1_small_beta_block(
            'block1', base_depth=depth_func(64), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block2', base_depth=depth_func(128), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block3', base_depth=depth_func(256), num_units=2, stride=2),
        resnet_v1_utils.Block('block4', BasicBlock, block4_args)]
    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        is_training=is_training,
                        global_pool=global_pool,
                        base_only=base_only,
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


def ResNetV1Beta_18(num_classes=None,
                    is_training=None,
                    global_pool=False,
                    base_only=False,
                    min_base_depth=8,
                    depth_multiplier=1,
                    root_depth_multiplier=0.25,
                    output_stride=None,
                    extract_blocks=None,
                    multi_grid=None,
                    name='resnet_v1_18'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID_RESNET_18
    elif len(multi_grid) != 2:
        raise ValueError("Expect multi_grid to have length 2.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    block4_args = []
    for rate in multi_grid:
        block4_args.append({'depth': depth_func(512), 'stride': 1, 'unit_rate': rate})

    blocks = [
        ResNet_V1_small_beta_block(
            'block1', base_depth=depth_func(64), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block2', base_depth=depth_func(128), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block3', base_depth=depth_func(256), num_units=2, stride=2),
        resnet_v1_utils.Block('block4', BasicBlock, block4_args)]
    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        is_training=is_training,
                        global_pool=global_pool,
                        base_only=base_only,
                        root_block=resnet_v1_utils.root_block(
                            root_depth_multiplier),
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


def ResNetV1_50(num_classes=None,
                is_training=None,
                base_only=False,
                global_pool=False,
                min_base_depth=8,
                depth_multiplier=1,
                output_stride=None,
                extract_blocks=None,
                multi_grid=None,
                name='resnet_v1_50'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=6, stride=2),
        resnet_v1_utils.Block('block4', Bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        is_training=is_training,
                        base_only=base_only,
                        global_pool=global_pool,
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


def ResNetV1Beta_50(num_classes=None,
                    is_training=None,
                    base_only=False,
                    global_pool=False,
                    min_base_depth=8,
                    depth_multiplier=1,
                    output_stride=None,
                    extract_blocks=None,
                    multi_grid=None,
                    name='resnet_v1_50'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=6, stride=2),
        resnet_v1_utils.Block('block4', Bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        is_training=is_training,
                        base_only=base_only,
                        global_pool=global_pool,
                        root_block=resnet_v1_utils.root_block(),
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


def ResNetV1_101(num_classes=None,
                 is_training=None,
                 base_only=False,
                 global_pool=False,
                 min_base_depth=8,
                 depth_multiplier=1,
                 output_stride=None,
                 extract_blocks=None,
                 multi_grid=None,
                 name='resnet_v1_101'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=23, stride=2),
        resnet_v1_utils.Block('block4', Bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        base_only=base_only,
                        is_training=is_training,
                        global_pool=global_pool,
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


def ResNetV1Beta_101(num_classes=None,
                     is_training=None,
                     base_only=False,
                     global_pool=False,
                     min_base_depth=8,
                     depth_multiplier=1,
                     output_stride=None,
                     extract_blocks=None,
                     multi_grid=None,
                     name='resnet_v1_101'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=23, stride=2),
        resnet_v1_utils.Block('block4', Bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        is_training=is_training,
                        base_only=base_only,
                        global_pool=global_pool,
                        root_block=resnet_v1_utils.root_block(),
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


def ResNetV1_152(num_classes=None,
                 is_training=None,
                 base_only=False,
                 global_pool=False,
                 min_base_depth=8,
                 depth_multiplier=1,
                 output_stride=None,
                 extract_blocks=None,
                 name='resnet_v1_152'):
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=8, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=36, stride=2),
        ResNet_V1_beta_block(
            'block4', base_depth=depth_func(512), num_units=3, stride=1)]
    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        is_training=is_training,
                        base_only=base_only,
                        global_pool=global_pool,
                        root_block=None,
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


def ResNetV1_200(num_classes=None,
                 is_training=None,
                 base_only=False,
                 global_pool=False,
                 min_base_depth=8,
                 depth_multiplier=1,
                 output_stride=None,
                 extract_blocks=None,
                 name='resnet_v1_200'):
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=24, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=36, stride=2),
        ResNet_V1_beta_block(
            'block4', base_depth=depth_func(512), num_units=3, stride=1)]

    return ResNetV1Beta(blocks=blocks,
                        num_classes=num_classes,
                        is_training=is_training,
                        base_only=base_only,
                        global_pool=global_pool,
                        output_stride=output_stride,
                        extract_blocks=extract_blocks,
                        name=name)


if __name__ == '__main__':
    data = tf.ones((2, 224, 224, 3))
    # with lib.engine.arg_scope(resnet_v1_utils.resnet_arg_scope()):
    #     model = ResNet_V1_101_beta((2, 224, 224, 3), num_classes=1000, is_training=True,
    #                                global_pool=True)
    # print(model(data))
    # for nodes in getattr(model, '_nodes_in_depth'):
    #     for node in nodes:
    #         for weight in node.downstream_layer.weights:
    #             print(weight)
    extract_blocks = ['pool1', 'block1/unit_2/lite_bottleneck_v1/add',
                      'block2/unit_2/lite_bottleneck_v1/add',
                      'block3/unit_2/lite_bottleneck_v1/add',
                      'block4/unit_2/lite_bottleneck_v1/add']
    model = ResNetV1_18(base_only=False)
    print(model(data))
