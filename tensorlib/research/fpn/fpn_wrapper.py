import tensorlib as lib
from tensorflow.python.ops import image_ops


class FPNWrapper(lib.engine.Network):

    def __init__(self,
                 network,
                 extract_blocks: (list, tuple),
                 depth=256,
                 dropout_rate=None,
                 **kwargs):
        super(FPNWrapper, self).__init__(**kwargs)
        assert isinstance(network, (lib.engine.Network, lib.engine.LayerList))
        self.dropout_rate = dropout_rate
        self.network = network
        self.extract_blocks = extract_blocks
        self.conv1x1 = lib.contrib.Conv2D(out_channels=depth, kernel_size=1)
        self.laterals = [lib.contrib.Conv2D(out_channels=depth, kernel_size=1)] * len(extract_blocks)
        self.top_downs = [lib.contrib.Conv2D(out_channels=depth, kernel_size=3)] * len(extract_blocks)
        self.add = lib.layers.ElementWise('add')
        if dropout_rate:
            self.dropout = lib.layers.Dropout(rate=dropout_rate)

    def forward(self, inputs):
        ret = []
        with lib.hooks.ExtractHook(
                prefix=self.network.name,
                in_names=None,
                out_names=self.extract_blocks) as hook:
            outputs = self.network(inputs)
        _, nets = hook.get_extract()
        outputs = self.conv1x1(outputs)
        for i, net in enumerate(reversed(nets)):
            spatial = lib.engine.int_shape(net)[1:-1]
            if spatial[0] != lib.engine.int_shape(outputs)[1]:
                outputs = lib.layers.Lambda(
                    image_ops.resize_bilinear, arguments=dict(
                        size=spatial, align_corners=True))(outputs)
            branch = self.laterals[i](net)
            outputs = self.add(outputs, branch)
            outputs = self.top_downs[i](outputs)
            ret.append(outputs)
        if self.dropout_rate:
            outputs = self.dropout(outputs)
        return outputs, ret
