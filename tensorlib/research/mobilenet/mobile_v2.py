import tensorlib as lib
from tensorlib.research.mobilenet.mobile_v2_utils import mobile_v2_conv, mobile_v2_blocks, expand_input_by_factor

"""                      
                     endpoints
---------------------------------------
block_0
    ----------------------------- 
    'convolution/output'
    -----------------------------
block_1
    -----------------------------
    'bottle_neck/projection_output'                   
    'bottle_neck/depthwise_output'
    'bottle_neck/output'
    -----------------------------
block_2
    -----------------------------
    'bottle_neck_1/expansion_output'
    'bottle_neck_1/projection_output'
    'bottle_neck_1/depthwise_output'
    'bottle_neck_1/output'
    'bottle_neck_2/expansion_output'
    'bottle_neck_2/projection_output'
    'bottle_neck_2/depthwise_output'
    'bottle_neck_2/output'
    -----------------------------
block_3
    -----------------------------
    'bottle_neck_3/expansion_output'
    'bottle_neck_3/projection_output'
    'bottle_neck_3/depthwise_output'
    'bottle_neck_3/output'
    'bottle_neck_4/expansion_output'
    'bottle_neck_4/projection_output'
    'bottle_neck_4/depthwise_output'
    'bottle_neck_4/output'
    'bottle_neck_5/expansion_output'
    'bottle_neck_5/projection_output'
    'bottle_neck_5/depthwise_output'
    'bottle_neck_5/output'
    -----------------------------
block_4
    -----------------------------
    'bottle_neck_6/expansion_output'
    'bottle_neck_6/projection_output'
    'bottle_neck_6/depthwise_output'
    'bottle_neck_6/output'
    'bottle_neck_7/expansion_output'
    'bottle_neck_7/projection_output'
    'bottle_neck_7/depthwise_output'
    'bottle_neck_7/output'
    'bottle_neck_8/expansion_output'
    'bottle_neck_8/projection_output'
    'bottle_neck_8/depthwise_output'
    'bottle_neck_8/output'
    'bottle_neck_9/expansion_output'
    'bottle_neck_9/projection_output'
    'bottle_neck_9/depthwise_output'
    'bottle_neck_9/output'
    -----------------------------
block_5
    -----------------------------
    'bottle_neck_10/expansion_output'
    'bottle_neck_10/projection_output'
    'bottle_neck_10/depthwise_output'
    'bottle_neck_10/output'
    'bottle_neck_11/expansion_output'
    'bottle_neck_11/projection_output'
    'bottle_neck_11/depthwise_output'
    'bottle_neck_11/output'
    'bottle_neck_12/expansion_output'
    'bottle_neck_12/projection_output'
    'bottle_neck_12/depthwise_output'
    'bottle_neck_12/output'
    -----------------------------
block_6
    -----------------------------
    'bottle_neck_13/expansion_output'
    'bottle_neck_13/projection_output'
    'bottle_neck_13/depthwise_output'
    'bottle_neck_13/output'
    'bottle_neck_14/expansion_output'
    'bottle_neck_14/projection_output'
    'bottle_neck_14/depthwise_output'
    'bottle_neck_14/output'
    'bottle_neck_15/expansion_output'
    'bottle_neck_15/projection_output'
    'bottle_neck_15/depthwise_output'
    'bottle_neck_15/output'
    -----------------------------
block_7
    -----------------------------
    'bottle_neck_16/expansion_output'
    'bottle_neck_16/projection_output'
    'bottle_neck_16/depthwise_output'
    'bottle_neck_16/output'
    -----------------------------
block_8
    -----------------------------
    'convolution_1/output'
    -----------------------------
---------------------------------------
"""
BLOCK_DEF = [
        mobile_v2_conv('block_0', kernel_size=(3, 3), stride=2, out_channels=32),
        mobile_v2_blocks('block_1', stride=1, num_units=1, out_channels=16,
                         expansion_factor=expand_input_by_factor(1, divisible_by=1)),
        mobile_v2_blocks('block_2', stride=2, num_units=2, out_channels=24),
        mobile_v2_blocks('block_3', stride=2, num_units=3, out_channels=32),
        mobile_v2_blocks('block_4', stride=2, num_units=4, out_channels=64),
        mobile_v2_blocks('block_5', stride=1, num_units=3, out_channels=96),
        mobile_v2_blocks('block_6', stride=2, num_units=3, out_channels=160),
        mobile_v2_blocks('block_7', stride=1, num_units=1, out_channels=320),
        mobile_v2_conv('block_8', kernel_size=(1, 1), stride=1, out_channels=1280)]


class MobileNetV2(lib.engine.Network):
    def __init__(self,
                 in_channels,
                 num_classes=1001,
                 endpoints=None,
                 prediction_fn=None,
                 base_only=False,
                 output_stride=None,
                 explicit_padding=False,
                 min_depth=None,
                 divisible_by=None,
                 multiplier=1.0,
                 block_def=None,
                 **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        self.bottlenecks = lib.LayerList()
        self.num_classes = num_classes
        self.prediction_fn = prediction_fn
        if endpoints is not None:
            if not isinstance(endpoints, list):
                raise TypeError("Expected type of endpoint is list, but given %s" % str(endpoints))
        self.endpoints = endpoints
        self._endpoints = {}

        if not base_only and output_stride is not None and output_stride != 32:
            raise ValueError("As the `base_only` is set to `False`, `output_stride` can only be 32, "
                             "but given %d." % output_stride)
        self.base_only = base_only

        if output_stride is not None and output_stride not in [8, 16, 32]:
            raise ValueError('Only allowed output_stride values are 8, 16, 32.')

        depth_args = {}
        if min_depth is not None:
            depth_args['min_depth'] = min_depth
        if divisible_by is not None:
            depth_args['divisible_by'] = divisible_by
        if multiplier <= 0:
            raise ValueError('`multiplier` is not greater than zero.')

        if output_stride is not None:
            if output_stride == 0 or (output_stride > 1 and output_stride % 2):
                raise ValueError('Output stride must be None, 1 or a multiple of 2.')

        current_stride = 1
        rate = 1
        block_def = block_def if block_def else BLOCK_DEF
        in_channels = in_channels
        with lib.arg_scope([lib.layers.BatchNorm], trainable=self.trainable):
            for block in block_def:
                for i, unit in enumerate(block.args):
                    params = dict(unit)
                    block_stride = params.get('stride', 1)
                    params['in_channels'] = in_channels
                    params['out_channels'] = params.pop('multiplier_func')(multiplier, **depth_args)
                    in_channels = params["out_channels"]
                    if output_stride is not None and current_stride == output_stride:
                        layer_stride = 1
                        layer_rate = rate
                        rate *= block_stride
                    else:
                        layer_stride = block_stride
                        layer_rate = rate
                        current_stride *= block_stride
                    params['stride'] = layer_stride
                    if layer_rate > 1:
                        if tuple(params.get('kernel_size', [])) != (1, 1):
                            params['rate'] = layer_rate
                    if explicit_padding:
                        params['explicit_padding'] = explicit_padding
                    self.bottlenecks.append(block.object(**params))
        self.logit_layer = lib.layers.Conv2D(out_channels=self.num_classes, kernel_size=(1, 1),
                                             activation=None, name="conv2d_1x1")

    @classmethod
    def _get_endpoints(cls, bottleneck):
        endpoints = {}
        if hasattr(bottleneck, "endpoints"):
            for key, value in bottleneck.endpoints.items():
                endpoints[bottleneck.name + "/" + str(key)] = value
        return endpoints

    def _check_return_endpoints(self):
        tensors = []
        if self.endpoints is not None:
            for endpoint in self.endpoints:
                try:
                    tensors.append(self._endpoints[endpoint])
                except KeyError:
                    raise RuntimeError("Got unknown endpoint %s" % str(endpoint))
        return tensors

    def forward(self, inputs):
        net = inputs
        for bottleneck in self.bottlenecks:
            net = bottleneck(net)
            self._endpoints.update(self._get_endpoints(bottleneck))
        return_points = self._check_return_endpoints()
        if return_points:
            return return_points
        if not self.base_only:
            shape = lib.engine.int_shape(net)
            kernel_size = (shape[1], shape[2])
            avg_pool = lib.layers.AvgPool2D(kernel_size=kernel_size)
            self.add_layer('avg_pool', avg_pool)
            net = avg_pool(net)
            net = self.logit_layer(net)
            net = lib.layers.squeeze(net, axis=[1, 2], activation=self.prediction_fn)
        return net
