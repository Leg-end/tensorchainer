import tensorflow as tf
import tensorlib as lib
import keras
tf.random.set_random_seed(666)


class LayerTest(tf.test.TestCase):

    def test_k_conv1d(self):
        conv = keras.layers.Conv1D(
            3, 7, strides=2,
            kernel_initializer='ones',
            padding='same')
        outputs = conv(tf.random.normal((1, 8, 3)))
        print(outputs)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(outputs))

    def test_conv1d(self):
        conv = lib.contrib.Conv1D(
            out_channels=3, kernel_size=7,
            strides=2, name='conv1',
            kernel_initializer='ones',
            normalizer_fn=None)
        outputs = conv(tf.random.normal((1, 8, 3)))
        print(outputs)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(outputs))

    def test_convolution(self):
        inputs = lib.Input(input_shape=(7, 7, 3), batch_size=2)
        outputs = lib.layers.Conv2D(3, 3, strides=2)(inputs)
        print(outputs)

    def test_batch_normal(self):
        inputs = lib.Input(input_shape=(7, 7, 3), batch_size=2)
        outputs = lib.layers.Conv2D(3, 3, padding='VALID')(inputs)
        bn = lib.layers.BatchNorm()
        # bn.train()
        outputs = bn(outputs)
        writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        writer.close()
        print(bn.trainable, True)
        self.assertListEqual(list(lib.engine.int_shape(outputs)), [2, 5, 5, 3])
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(outputs))

    def test_lambda(self):
        inputs = [lib.Input((2,)), lib.Input((2,))]
        outputs = lib.layers.Lambda(tf.add, name='add')(*inputs)
        self.assertListEqual(list(lib.engine.int_shape(outputs)), [1, 2])
        with self.cached_session() as sess:
            print(sess.run(outputs))

    def test_dropout(self):
        inputs = lib.Input((5, 5))
        net = lib.layers.Dropout(0.5)
        net.eval()
        outputs = net(inputs)
        self.assertListEqual(list(lib.engine.int_shape(outputs)), [1, 5, 5])
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(outputs))
