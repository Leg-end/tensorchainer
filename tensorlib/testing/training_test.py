import tensorflow as tf
import tensorlib as lib
import keras


class TrainingTest(tf.test.TestCase):

    def test_estimator_style_fit(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        # x_train = np.random.random((1000, 20))
        # y_train = np.random.randint(2, size=(1000, 1))
        # x_test = np.random.random((100, 20))
        # y_test = np.random.randint(2, size=(100, 1))
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        class MNIST(lib.training.Model):
            def __init__(self, *args, **kwargs):
                super(MNIST, self).__init__(*args, **kwargs)
                self.flatten = lib.layers.Flatten()
                self.fc1 = lib.layers.Dense(units=128, activation='relu')
                self.fc2 = lib.layers.Dense(units=10)

            def forward(self, inputs):
                x = self.flatten(inputs)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        def model_fn(_model, inputs, labels):
            outputs = _model(inputs)
            loss = lib.training.SparseCategoricalCrossEntropy(from_logits=True)(labels, outputs)
            metric = lib.training.SparseCategoricalAccuracy()(labels, outputs)
            return lib.training.EstimatorSpec(outputs=outputs,
                                              loss=loss,
                                              metrics=[metric])

        model = MNIST()
        model.train()
        model.compile(model_fn=model_fn,
                      optimizer=tf.train.AdamOptimizer(),
                      checkpoint_dir='./test_ckpt',
                      session_cfg=dict(per_process_gpu_memory_fraction=0.4))
        model.fit(train_images, train_labels,
                  test_images, test_labels,
                  epochs=10)
        # writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        # writer.close()

    def test_graph_model_estimator_style_fit(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        # x_train = np.random.random((1000, 20))
        # y_train = np.random.randint(2, size=(1000, 1))
        # x_test = np.random.random((100, 20))
        # y_test = np.random.randint(2, size=(100, 1))
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        class MNIST(lib.training.Model):
            def __init__(self, *args, **kwargs):
                super(MNIST, self).__init__(*args, **kwargs)
                self.flatten = lib.layers.Flatten()
                self.fc1 = lib.layers.Dense(units=128, activation='relu')
                self.fc2 = lib.layers.Dense(units=10)

            def forward(self, inputs):
                x = self.flatten(inputs)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        def model_fn(_model, x, y):
            loss = lib.training.SparseCategoricalCrossEntropy(from_logits=True)(y, _model.outputs[0])
            metric = lib.training.SparseCategoricalAccuracy()(y, _model.outputs[0])
            return lib.training.EstimatorSpec(outputs=_model.outputs,
                                              loss=loss,
                                              metrics=[metric])

        inputs = lib.Input(input_shape=(28, 28))
        outputs = MNIST()(inputs)
        model = lib.training.Model(inputs=inputs, outputs=outputs, name='mnist/')
        model.train()
        model.compile(model_fn=model_fn,
                      optimizer=tf.train.AdamOptimizer(),
                      checkpoint_dir='./test_ckpt',
                      session_cfg=dict(per_process_gpu_memory_fraction=0.4))
        model.fit(train_images, train_labels,
                  test_images, test_labels,
                  epochs=10)

    def test_keras_style_fit(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        # x_train = np.random.random((1000, 20))
        # y_train = np.random.randint(2, size=(1000, 1))
        # x_test = np.random.random((100, 20))
        # y_test = np.random.randint(2, size=(100, 1))
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        class MNIST(lib.training.Model):
            def __init__(self, *args, **kwargs):
                super(MNIST, self).__init__(*args, **kwargs)
                self.flatten = lib.layers.Flatten()
                self.fc1 = lib.layers.Dense(units=128, activation='relu')
                self.fc2 = lib.layers.Dense(units=10)

            def forward(self, inputs):
                x = self.flatten(inputs)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = MNIST()
        model.train()
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=lib.training.SparseCategoricalCrossEntropy(from_logits=True),
                      metrics=['sce_acc'],
                      checkpoint_dir='./test_ckpt',
                      session_cfg=dict(per_process_gpu_memory_fraction=0.4))
        model.fit(train_images, train_labels,
                  test_images, test_labels,
                  epochs=10)

    def test_graph_model_keras_style_fit(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        # x_train = np.random.random((1000, 20))
        # y_train = np.random.randint(2, size=(1000, 1))
        # x_test = np.random.random((100, 20))
        # y_test = np.random.randint(2, size=(100, 1))
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        class MNIST(lib.training.Model):
            def __init__(self, *args, **kwargs):
                super(MNIST, self).__init__(*args, **kwargs)
                self.flatten = lib.layers.Flatten()
                self.fc1 = lib.layers.Dense(units=128, activation='relu')
                self.fc2 = lib.layers.Dense(units=10)

            def forward(self, inputs):
                x = self.flatten(inputs)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        inputs = lib.Input(input_shape=(28, 28))
        outputs = MNIST()(inputs)
        model = lib.training.Model(inputs=inputs, outputs=outputs, name='mnist/')
        model.train()
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=lib.training.SparseCategoricalCrossEntropy(from_logits=True),
                      metrics=['sce_acc'],
                      checkpoint_dir='./test_ckpt',
                      session_cfg=dict(per_process_gpu_memory_fraction=0.4))
        model.fit(train_images, train_labels,
                  test_images, test_labels,
                  epochs=10)

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
