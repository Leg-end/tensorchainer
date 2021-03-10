import tensorflow as tf
from tensorlib.layers.dense import Dense
from tensorlib.engine.input_layer import Input
from tensorlib.engine.sequential import Sequential
from tensorlib.engine.network import Network, graph_scope
import tensorlib as lib
import time


def demo_init_model():
    net = Sequential()
    net.append(Dense(10, name='dense1'))
    net.append(Dense(10, name='dense2'))
    net.append(Dense(1, name='dense3'))
    return net


def demo_graph_model():
    net = demo_init_model()
    with tf.name_scope('network') as scope:
        inputs = Input((2,), batch_size=2)
        outputs = net(inputs)
    net = Network(inputs=inputs, outputs=outputs, name=scope)
    return net


def demo_loop_graph_model():
    with tf.name_scope('network_top') as scope:
        net = demo_graph_model()
        inputs = Input((2,), batch_size=2)
        inputs = Dense(2, name='dense')(inputs)
        outputs = net(inputs)
    net = Network(inputs=inputs, outputs=outputs, name=scope)
    return net


def to_graph_network():
    with graph_scope('network_graph', values=[Input(input_shape=(2,))]) as handler:
        inputs = handler.inputs
        net = Dense(10, name='dense1')(inputs)
        net = Dense(10, name='dense2')(net)
        net = Dense(1, name='dense3')(net)
        handler.outputs = net
    return net


def create_model(input_shape):
    with tf.name_scope('model') as scope:
        model_inputs = lib.Input(batch_input_shape=input_shape)
        net = lib.research.MobileNetV2(3, base_only=True)(model_inputs)
        classes = lib.layers.Conv2D(out_channels=10, kernel_size=(1, 13))(net)
        pattern = lib.layers.Dense(num_units=24)(lib.layers.flatten(classes))
        pattern = lib.layers.reshape(pattern, (-1, 1, 1, 24))
        pattern = lib.layers.tile(pattern, [1, 1, lib.engine.int_shape(net)[2], 1])
        net = lib.layers.concat(classes, pattern, axis=3)
        net = lib.layers.Conv2D(out_channels=10, kernel_size=1)(net)
        model_outputs = lib.layers.squeeze(net, axis=1)
    return lib.engine.Network(inputs=model_inputs, outputs=model_outputs, name=scope)


class StructureTest(tf.test.TestCase):

    def test_layer(self):
        inputs = Input((2,))
        net = Dense(10, name='dense')
        outputs = net(inputs)
        self.assertListEqual(outputs.get_shape().as_list(), [1, 10])

    def test_weight(self):
        net = demo_init_model()
        net.train()
        inputs = Input((2,), batch_size=2)
        outputs = net(inputs)
        for w in net.trainable_weights:
            print(w)

    def test_init_model(self):
        net = demo_init_model()
        inputs = Input((2,), batch_size=2)
        outputs = net(inputs)
        node = getattr(outputs, '_anchor')[0]
        self.assertEqual(node.layer.name, 'dense3')
        self.assertListEqual(outputs.get_shape().as_list(), [2, 1])

    def test_graph_model(self):
        start = time.time()
        net = demo_graph_model()
        inputs = Input(input_shape=(2,), batch_size=2)
        outputs = net(inputs)
        print(time.time() - start)
        node = getattr(outputs, '_anchor')[0]
        self.assertEqual(node.layer.name, 'network/')
        self.assertListEqual(outputs.get_shape().as_list(), [2, 1])

    def test_loop_graph_model(self):
        start = time.time()
        net = demo_loop_graph_model()
        inputs = Input((2,), batch_size=2)
        outputs = net(inputs)
        print(time.time() - start)
        node = getattr(outputs, '_anchor')[0]
        self.assertEqual(node.layer.name, 'network_top/')
        self.assertListEqual(outputs.get_shape().as_list(), [2, 1])
        writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        writer.close()

    def test_to_graph(self):
        start = time.time()
        outputs = to_graph_network()
        print(time.time() - start)
        node = getattr(outputs, '_anchor')[0]
        self.assertEqual(node.layer.name, 'network_graph/')
        self.assertListEqual(outputs.get_shape().as_list(), [1, 1])
        writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        writer.close()

    def test_state(self):
        net = demo_init_model()
        net.train()
        for layer in net.layers():
            print(layer.name, layer.trainable)
        print('='*10)
        net = demo_graph_model()
        net.train()
        for layer in net.layers():
            print(layer.name, layer.trainable)
        print('=' * 10)
        net = demo_loop_graph_model()
        net.train()
        for layer in net.layers():
            print(layer.name, layer.trainable)

    def test_graph_model_v2(self):
        model = create_model((1, 24, 94, 3))
        outputs = model(tf.ones((1, 24, 94, 3)))
        node = getattr(outputs, '_anchor')[0]
        print(node.layer.name)
        for n in node.in_degree:
            if n:
                print(n.layer.name)
            else:
                print('None')
        print(outputs)
        writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        writer.close()

    def test_nan(self):
        import numpy as np
        ph = tf.placeholder(dtype=tf.float32, shape=(1, 2))
        inputs = ph * np.nan
        with lib.hooks.NumericHook():
            outputs = lib.layers.Dense(units=3)(inputs)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            i, o = sess.run([inputs, outputs], feed_dict={ph: np.ones((1, 2))})
        print(i)
        print(o)

    def test_num_scale(self):
        import numpy as np
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 4))
        with lib.hooks.NumericScaleHook():
            outputs = lib.layers.Dense(units=3)(inputs)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            i, o = sess.run([inputs, outputs], feed_dict={inputs: np.random.randn(10, 4)})
        print(i)
        print(o)

