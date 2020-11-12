import tensorflow as tf
import tensorlib as lib


class RNNTest(tf.test.TestCase):

    def test_rnn(self):
        inputs = lib.Input(batch_input_shape=(2, 10, 28))
        rnn = lib.layers.RNN(10)
        outputs = rnn(inputs)
        node = getattr(outputs, '_anchor')[0]
        self.assertEqual(node.layer.name, rnn.name)
        self.assertListEqual(list(lib.engine.int_shape(outputs)), [2, 10])
        # writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        # writer.close()
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(outputs))
