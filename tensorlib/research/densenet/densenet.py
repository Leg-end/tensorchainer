import tensorlib as lib
import math


class DenseUnit(lib.Sequential):
    def __init__(self,
                 growth_rate,
                 bn_factor=4,
                 drop_rate=0.,
                 **kwargs):
        super(DenseUnit, self).__init__(**kwargs)
        self.add_layer(lib.contrib.Conv2D(out_channels=bn_factor * growth_rate,
                                          kernel_size=1))
        self.add_layer(lib.contrib.Conv2D(out_channels=growth_rate,
                                          kernel_size=3))
        if drop_rate > 0.:
            self.add_layer(lib.layers.Dropout(rate=drop_rate))

    def forward(self, inputs):
        outputs = super(DenseUnit, self).forward(inputs)
        outputs = lib.layers.concat([inputs, outputs],
                                    axis=self[0][0].data_format.find('C'))
        return outputs


class DenseBlock(lib.Network):
    def __init__(self,
                 growth_rate,
                 num_layer,
                 bn_factor=4,  # bottleneck
                 drop_rate=0.,
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.stack_layers = [DenseUnit(
            growth_rate, bn_factor, drop_rate)
            for _ in range(num_layer)]

    def forward(self, inputs):
        outputs = inputs
        for layer in self.stack_layers:
            outputs = layer(outputs)
        return outputs


def transition_layers(out_channels, compression):
    return lib.Sequential(lib.layers.BatchNorm(activation='relu'),
                          lib.layers.Conv2D(out_channels=math.floor(out_channels * compression),
                                            kernel_size=1),
                          lib.layers.AvgPool2D(kernel_size=2,
                                               padding='SAME'),
                          name='transition')


class DenseNet(lib.Network):

    def __init__(self,
                 growth_rate,
                 num_layers,
                 compression=0.5,
                 bn_factor=4,
                 drop_rate=0.,
                 num_classes=1000,
                 **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.conv1 = lib.contrib.Conv2D(out_channels=2 * growth_rate,
                                        kernel_size=7, strides=2)
        self.max_pool = lib.layers.MaxPool2D(kernel_size=3, strides=2)

        self.layer1 = DenseBlock(growth_rate, num_layers[0], bn_factor,
                                 drop_rate)
        self.trans1 = transition_layers(growth_rate, compression)
        self.layer2 = DenseBlock(growth_rate, num_layers[1], bn_factor,
                                 drop_rate)
        self.trans2 = transition_layers(growth_rate, compression)
        self.layer3 = DenseBlock(growth_rate, num_layers[2], bn_factor,
                                 drop_rate)
        self.trans3 = transition_layers(growth_rate, compression)
        self.layer4 = DenseBlock(growth_rate, num_layers[3], bn_factor,
                                 drop_rate)

        self.global_pool = lib.layers.GlobalAvgPool()
        self.flatten = lib.layers.Flatten()
        self.classifier = lib.layers.Dense(units=num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.max_pool(outputs)

        outputs = self.layer1(outputs)
        outputs = self.trans1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.trans2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.trans3(outputs)
        outputs = self.layer4(outputs)

        outputs = self.global_pool(outputs)
        outputs = self.flatten(outputs)
        outputs = self.classifier(outputs)
        return outputs


def _DenseNet(growth_rate, block_config, **kwargs):
    model = DenseNet(growth_rate, block_config, **kwargs)
    return model


def DenseNet121(**kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'densenet121'
    return _DenseNet(32, (6, 12, 24, 16), **kwargs)


def DenseNet161(**kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'densenet161'
    return _DenseNet(48, (6, 12, 36, 24), **kwargs)


def DenseNet169(**kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'densenet169'
    return _DenseNet(32, (6, 12, 32, 32), **kwargs)


def DenseNet201(**kwargs):
    if 'name' not in kwargs:
        kwargs['name'] = 'densenet201'
    return _DenseNet(32, (6, 12, 48, 32), **kwargs)


def densenet_arg_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        activation='relu',
        data_format='channels_last',
        use_batch_norm=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'data_format': data_format
    }

    with lib.engine.arg_scope(
            [lib.contrib.Conv2D],
            kernel_regularizer=lib.regularizers.l2(weight_decay),
            kernel_initializer='truncated_normal',
            activation=activation,
            data_format=data_format,
            normalizer=lib.layers.BatchNorm if use_batch_norm else None):
        with lib.engine.arg_scope([lib.layers.BatchNorm], **batch_norm_params):
            with lib.engine.arg_scope([lib.layers.MaxPool2D, lib.layers.Conv2D],
                                      padding='SAME', data_format=data_format) as arg_sc:
                return arg_sc


if __name__ == '__main__':
    import tensorflow as tf
    with lib.engine.arg_scope(densenet_arg_scope()):
        net = DenseNet121()
    print(net(tf.ones((1, 224, 224, 3))))
    writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph/model_graph", tf.get_default_graph())
    writer.close()
