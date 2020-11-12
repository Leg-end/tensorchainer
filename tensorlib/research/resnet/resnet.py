import tensorlib as lib
from tensorlib.research.resnet import resnet_util


class BasicBlock(lib.engine.Network):
    expansion = 1

    @lib.engine.add_arg_scope
    def __init__(self,
                 depth,
                 stride=1,
                 unit_rate=1,
                 rate=1,
                 downsample=None,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = resnet_util.conv2d_same(depth=depth,
                                             kernel_size=3,
                                             stride=1,
                                             rate=rate * unit_rate,
                                             name='conv1')
        self.conv2 = resnet_util.conv2d_same(depth=depth,
                                             kernel_size=3,
                                             stride=stride,
                                             rate=rate * unit_rate,
                                             activation=None,
                                             name='conv2')

        self.add = lib.layers.ElementWise('add', activation='relu')
        self.downsample = downsample

    def forward(self, inputs):
        identity = inputs
        residual = self.conv1(inputs)
        residual = self.conv2(residual)

        if self.downsample is not None:
            identity = self.downsample(identity)
        outputs = self.add(identity, residual)
        return outputs


class Bottleneck(lib.engine.Network):
    expansion = 4

    @lib.engine.add_arg_scope
    def __init__(self,
                 depth_bottleneck,
                 stride=1,
                 unit_rate=1,
                 rate=1,
                 downsample=None,
                 **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = lib.contrib.Conv2D(
            out_channels=depth_bottleneck,
            kernel_size=1,
            strides=1,
            name='conv1')
        self.conv2 = resnet_util.conv2d_same(depth=depth_bottleneck,
                                             kernel_size=3,
                                             stride=stride,
                                             rate=rate * unit_rate,
                                             name='conv2')
        self.conv3 = lib.contrib.Conv2D(
            out_channels=depth_bottleneck * self.expansion,
            kernel_size=1,
            strides=1,
            activation=None,
            name='conv3')

        self.add = lib.layers.ElementWise('add', activation='relu')
        self.downsample = downsample

    def forward(self, inputs):
        identity = inputs

        residual = self.conv1(inputs)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        if self.downsample is not None:
            identity = self.downsample(identity)
        outputs = self.add(identity, residual)
        return outputs


class ResNet(lib.engine.Network):

    def __init__(self,
                 block,
                 num_block,
                 block_strides=(1, 2, 2, 2),
                 num_classes=1000,
                 **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.channels = 64
        self.num_classes = num_classes
        assert len(block_strides) == 4
        self.block_strides = block_strides
        self.conv1 = lib.contrib.Conv2D(64, kernel_size=7, strides=2)
        self.max_pool = lib.layers.MaxPool2D(kernel_size=3, strides=2, padding='SAME')
        self.layer1 = self._make_layer(block, 64, num_block[0], strides=block_strides[0])
        self.layer2 = self._make_layer(block, 128, num_block[1], strides=block_strides[1])
        self.layer3 = self._make_layer(block, 256, num_block[2], strides=block_strides[2])
        self.layer4 = self._make_layer(block, 512, num_block[3], strides=block_strides[3])
        self.classifier = lib.Sequential(lib.layers.GlobalAvgPool(),
                                         lib.layers.Conv2D(num_classes, 1),
                                         lib.layers.Flatten(),
                                         name='classifier')

    def _make_layer(self, block, out_channels, num_block, strides=1):
        downsample = None
        if self.channels != out_channels * block.expansion:
            downsample = lib.contrib.Conv2D(out_channels * block.expansion,
                                            kernel_size=1, strides=strides)
        elif strides != 1:
            downsample = lib.layers.MaxPool2D(kernel_size=1, strides=strides)
        blocks = [block(out_channels, strides, downsample=downsample)]
        for i in range(1, num_block):
            blocks.append(block(out_channels))
        return lib.Sequential(*blocks)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.max_pool(outputs)

        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.classifier(outputs)

        return outputs


def resnet18(num_classes=1000, **kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'resnetv1_18'
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   num_classes=num_classes, **kwargs)
    return model


def resnet34(num_classes=1000, **kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'resnetv1_34'
    model = ResNet(BasicBlock, [3, 4, 6, 3],
                   num_classes=num_classes, **kwargs)
    return model


def resnet50(num_classes=1000, **kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'resnetv1_50'
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   num_classes=num_classes, **kwargs)
    return model


def resnet101(num_classes=1000, **kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'resnetv1_101'
    model = ResNet(BasicBlock, [3, 4, 23, 3],
                   num_classes=num_classes, **kwargs)
    return model


def resnet152(num_classes=1000, **kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'resnetv1_152'
    model = ResNet(BasicBlock, [3, 8, 36, 3],
                   num_classes=num_classes, **kwargs)
    return model
