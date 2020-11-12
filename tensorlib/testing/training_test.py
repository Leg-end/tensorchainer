import tensorflow as tf
import tensorlib as lib
import keras


class TrainingTest(tf.test.TestCase):

    def test_fit(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        # x_train = np.random.random((1000, 20))
        # y_train = np.random.randint(2, size=(1000, 1))
        # x_test = np.random.random((100, 20))
        # y_test = np.random.randint(2, size=(100, 1))
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        network = lib.Sequential(lib.layers.Flatten(),
                                 lib.layers.Dense(units=128, activation='relu'),
                                 lib.layers.Dense(units=10))
        network.train()
        def activation(x): return tf.reduce_max(tf.nn.softmax(x, axis=-1), axis=-1, keepdims=True)
        model = lib.training.Model(network=network)
        model.compile(loss=lib.training.SparseCategoricalCrossEntropy(from_logits=True),
                      optimizer=tf.train.AdamOptimizer(),
                      activations=activation,
                      metrics=['sce_acc'])
        # model.fit(train_images, train_labels,
        #           test_images, test_labels,
        #           epochs=1)
        model.evaluate(test_images, test_labels)
        # writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        # writer.close()

    def test_exe(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        # x_train = np.random.random((1000, 20))
        # y_train = np.random.randint(2, size=(1000, 1))
        # x_test = np.random.random((100, 20))
        # y_test = np.random.randint(2, size=(100, 1))
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        def model_fn(features, labels):
            network = lib.Sequential(lib.layers.Flatten(),
                                     lib.layers.Dense(units=128, activation='relu'),
                                     lib.layers.Dense(units=10))
            network.train()
            outputs = network(features)
            loss = lib.training.SparseCategoricalCrossEntropy(from_logits=True)(
                labels, outputs)
            metrics = [lib.training.SparseCategoricalAccuracy()(labels, outputs)]
            params = list(network.trainable_weights)
            return lib.training.ExecutorSpec(
                outputs=outputs,
                loss=loss,
                metrics=metrics,
                params=params)

        exe = lib.training.Executor(model_fn)
        exe.compile(optimizer=tf.train.AdamOptimizer(),
                    checkpoint_dir='./test_ckpt',
                    per_process_gpu_memory_fraction=0.4)
        exe.fit(train_images, train_labels,
                test_images, test_labels,
                epochs=6)
        # exe.evaluate(test_images, test_labels)
