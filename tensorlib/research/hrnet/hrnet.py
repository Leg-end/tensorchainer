from tensorlib.research.hrnet.hrnet_utils import *


def res_block(net, block):
    with tf.name_scope(block.scope, 'block', [net]):
        for i, unit in enumerate(block.args):
            point_name = 'unit_%d' % (i + 1)
            with tf.name_scope(point_name, values=[net]):
                net = block.unit_fn(net, **unit)
    return net


def hr_module(nets, depths, block_type='bottleneck', name='hr_module'):
    with tf.name_scope(name, values=nets):
        for i, depth in enumerate(depths):
            block = BLOCKS[block_type](scope='block_%d' % (i + 1), base_depth=depth, num_units=4, stride=1)
            nets[i] = res_block(nets[i], block)
    return nets


def hr_stage(scope, depths, num_modules, block_type='lite_bottleneck'):
    return HRStage(scope, hr_module, [{
        'depths': depths,
        'block_type': block_type
    }] * num_modules)


def HRNet_base(stages,
               head_net,
               input_shape=(224, 224, 3),
               is_training=None,
               name='hrnet'):
    with tf.name_scope(name) as scope:
        inputs = lib.engine.Input(input_shape=input_shape)
        if is_training is not None:
            arg_scope = lib.engine.arg_scope(
                [lib.layers.BatchNorm], trainable=is_training)
        else:
            arg_scope = lib.engine.arg_scope([])
        with arg_scope:
            net = inputs
            net = lib.contrib.Conv2D(out_channels=64, kernel_size=3,
                                     stride=2)(net)
            net = lib.contrib.Conv2D(out_channels=64, kernel_size=3,
                                     stride=2)(net)
            nets = stack_stages(net, stages)
            nets = head_net(nets)
    return lib.engine.Network(inputs=inputs, outputs=nets, name=scope, trainable=is_training)


def HRNet_v1(stages=None,
             input_shape=(224, 224, 3),
             is_training=None,
             base_depth=32,
             name='hrnet_v1'):
    def _head_net(nets):
        return nets[0]

    if stages is None:
        stages = [
            hr_stage(
                'stage_1', depths=[64], num_modules=1, block_type='bottleneck'),
            hr_stage(
                'stage_2', depths=[base_depth, base_depth * 2], num_modules=1),
            hr_stage(
                'stage_3', depths=[base_depth, base_depth * 2, base_depth * 4], num_modules=4),
            hr_stage(
                'stage_4', depths=[base_depth, base_depth * 2, base_depth * 4, base_depth * 8], num_modules=3)]
    return HRNet_base(stages=stages,
                      head_net=_head_net,
                      input_shape=input_shape,
                      is_training=is_training,
                      name=name)


def HRNet_v2(stages=None,
             input_shape=(224, 224, 3),
             is_training=None,
             base_depth=32,
             num_classes=1001,
             name='hrnet_v2'):
    def _head_net(nets):
        size = lib.engine.int_shape(nets[0])[1: -1]
        for i in range(len(nets) - 1, -1):
            nets[i] = lib.layers.Lambda(lambda x: tf.image.resize_bilinear(
                x, size=size, align_corners=True), name='upsample')(nets[i])
        net = lib.layers.concat(*nets, axis=-1)
        net = lib.contrib.Conv2D(out_channels=num_classes,
                                 kernel_size=1)(net)
        return net

    if stages is None:
        stages = [
            hr_stage(
                'stage_1', depths=[64], num_modules=1, block_type='bottleneck'),
            hr_stage(
                'stage_2', depths=[base_depth, base_depth * 2], num_modules=1),
            hr_stage(
                'stage_3', depths=[base_depth, base_depth * 2, base_depth * 4], num_modules=4),
            hr_stage(
                'stage_4', depths=[base_depth, base_depth * 2, base_depth * 4, base_depth * 8], num_modules=3)]
    return HRNet_base(stages=stages,
                      head_net=_head_net,
                      input_shape=input_shape,
                      is_training=is_training,
                      name=name)


def HRNet_v2p(stages=None,
              input_shape=(224, 224, 3),
              is_training=None,
              base_depth=32,
              name='hrnet_v2p'):
    def _head_net(nets):
        size = lib.engine.int_shape(nets[0])[1: -1]
        num = len(nets)
        for i in range(num - 1, -1):
            nets[i] = lib.layers.Lambda(lambda x: tf.image.resize_bilinear(
                x, size=size, align_corners=True), name='upsample')(nets[i])
        net = lib.layers.concat(*nets, axis=-1)
        nets = [net]
        for _ in range(num - 1):
            net = lib.contrib.Conv2D(out_channels=net.shape[-1],
                                     kernel_size=3, stride=2)(net)
            nets.append(net)
        return nets

    if stages is None:
        stages = [
            hr_stage(
                'stage_1', depths=[64], num_modules=1, block_type='bottleneck'),
            hr_stage(
                'stage_2', depths=[base_depth, base_depth * 2], num_modules=1),
            hr_stage(
                'stage_3', depths=[base_depth, base_depth * 2, base_depth * 4], num_modules=4),
            hr_stage(
                'stage_4', depths=[base_depth, base_depth * 2, base_depth * 4, base_depth * 8], num_modules=3)]
    return HRNet_base(stages=stages,
                      head_net=_head_net,
                      input_shape=input_shape,
                      is_training=is_training,
                      name=name)


if __name__ == '__main__':
    a = tf.ones((2, 224, 224, 3))
    model = HRNet_v1()
    print(model)
    print(model(a))
    writer = tf.summary.FileWriter(r'D:\GeekGank\workspace\graph\model_graph', tf.get_default_graph())
    writer.close()
