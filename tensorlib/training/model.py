from tensorflow import train
from tensorflow.python import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops as fops
from tensorflow import logging
import numpy as np
import copy
import time
import os
from inspect import isgenerator
from scipy.sparse import issparse

from tensorlib.utils import to_list, slice_arrays, has_arg,\
    unpack_singleton, list_files, validate_kwargs
from tensorlib.engine import base_lib as F
from tensorlib.training.sessions.session import Function
from tensorlib.training import losses
from tensorlib.training import metrics as metric_module
from tensorlib.training import utils
from tensorlib.training import ProgressHook, CkPtSaverHook
from tensorlib.training.optimizers import Optimizer


class Model(object):

    @property
    def built(self):
        return self._built

    @property
    def is_compiled(self):
        return self._is_compiled

    @property
    def uses_learning_phase(self):
        return self._uses_learning_phase

    def __init__(self,
                 network,
                 **kwargs):
        logging.set_verbosity(logging.INFO)
        validate_kwargs(kwargs, {'model_dir',
                                 'save_checkpoints_steps',
                                 'save_summary_steps'})
        self.cfg = kwargs
        self.global_step = None
        self.network = network
        self.inputs = []
        self.input_names = []
        self.outputs = []
        self.output_names = []
        # Whether network has been called
        self._built = False
        self._is_compiled = False
        self._uses_learning_phase = False

        self.optimizer = None
        self.loss = None
        self.loss_weights = None
        self.loss_functions = None
        self.activations = None
        self.metrics = None

        self._feed_inputs = []
        self._feed_input_names = []
        self._feed_input_shapes = []

        self._function_kwargs = {}
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def _compile_args(self, args, tag, default=None):
        if isinstance(args, dict):
            ret = []
            for arg in args:
                if arg not in self.output_names:
                    raise ValueError("Unknown entry in %s dictionary: %s."
                                     "Only expected the following keys: %s"
                                     % (tag, str(arg), str(self.output_names)))
            for name in self.output_names:
                ret.append(args.get(name, default))
        else:
            args = to_list(args)
            if len(args) != len(self.outputs):
                raise ValueError("Mismatch length between %s and outputs"
                                 " with %d vs %d" % (tag, len(args), len(self.outputs)))
            ret = args
        return ret

    def _compile_loss(self):
        self.metric_names = ['loss']  # map with total_loss
        self.metric_tensors = []
        total_loss = 0.
        with ops.name_scope('loss'):
            for i in range(len(self.outputs)):
                if i in self._skip_target_indices:
                    continue
                loss_function = self._feed_loss_fns[i]
                target = self.targets[i]
                output = self.outputs[i]
                loss_weight = self.loss_weights[i]
                output_loss = loss_function(target, output)
                total_loss += loss_weight * output_loss
                if len(self.outputs) > 1:
                    self.metric_tensors.append(output_loss)
                    self.metric_names.append(self.output_names[i] + '_loss')
            reg_loss = fops.get_collection(fops.GraphKeys.REGULARIZATION_LOSSES)
            if reg_loss:
                total_loss += math_ops.add_n(reg_loss)
        self.total_loss = total_loss

    def _compile_metrics(self, metrics):
        # Must handle sparse situation carefully!
        def _compile_metric(m, loss_fn):
            if isinstance(loss_fn, losses.SparseCategoricalCrossEntropy):
                if m in {'accuracy', 'acc'}:
                    m = metric_module.SparseCategoricalAccuracy()
                    return m
            m = metric_module.get(m)
            return m

        if not metrics:
            nested_metrics = [[]] * len(self.outputs)
        elif isinstance(metrics, list):
            nested_metrics = [copy.copy(metrics)] * len(self.outputs)
        elif isinstance(metrics, dict):
            nested_metrics = [to_list(metrics.get(name, []))
                              for name in self.output_names]
        else:
            raise TypeError("Unexpected type of metrics: " + str(metrics))
        self.stateful_metrics = []
        self.stateful_metric_names = []
        with ops.name_scope('metrics'):
            for i in range(len(self.outputs)):
                if i in self._skip_target_indices:
                    continue
                target = self.targets[i]
                output = self.outputs[i]
                output_metrics = nested_metrics[i]
                loss_function = self.loss_functions[i]
                for metric in output_metrics:
                    metric = _compile_metric(metric, loss_function)
                    metric_name = metric.name if hasattr(
                        metric, 'name') else metric.__name__
                    with ops.name_scope(metric_name):
                        metric_result = metric(target, output)
                        if len(self.output_names) > 1:
                            metric_name = self.output_names[i] + '_' + metric_name
                        self.metric_names.append(metric_name)
                        self.metric_tensors.append(metric_result)
                        if isinstance(metric, metric_module.Metric):
                            self.stateful_metrics.append(metric)
                            self.stateful_metric_names.append(metric_name)

    def _compile_loss_functions(self, loss):
        loss = self._compile_args(loss, 'loss')
        self.loss_functions = [losses.get(name) for name in loss]
        self._skip_target_indices = [i for i, fn in enumerate(
            self.loss_functions) if fn is None]

    def _compile_target_tensors(self, targets):
        self.targets = []
        self._feed_targets = []
        self._feed_outputs = []
        self._feed_output_names = []
        self._feed_output_shapes = []
        self._feed_loss_fns = []
        targets = self._compile_args(targets, 'target_tensors')
        targets = [target if F.is_tensor(target) else None
                   for target in targets]
        for i in range(len(self.outputs)):
            if i in self._skip_target_indices:
                self.targets.append(None)
            else:
                name = self.output_names[i]
                output = self.outputs[i]
                shape = F.int_shape(output)
                target = targets[i]
                loss_fn = self.loss_functions[i]
                if target is None:
                    target = F.placeholder(
                        ndim=len(shape),
                        name=name + '_target',
                        sparse=F.is_sparse(output),
                        dtype=F.dtype(output))
                if F.is_placeholder(target):
                    self._feed_targets.append(target)
                    self._feed_outputs.append(output)
                    self._feed_output_names.append(name)
                    self._feed_loss_fns.append(loss_fn)
                    if not hasattr(loss_fn, '__name__') or loss_fn is None:
                        # If loss_fn is not as expected,
                        # then make no assumptions on output shape
                        self._feed_output_shapes.append(None)
                    else:
                        # output_shape should compatible to labels' shape
                        self._feed_output_shapes.append(
                            F.int_shape(self.activations[i](output)))
                self.targets.append(target)

    def _compile_activations(self, activations):
        def _linear(x):
            return x

        if activations is None:
            activations = [_linear] * len(self.outputs)
        else:
            activations = self._compile_args(activations, 'activations', default=_linear)
            assert all(callable(a) for a in activations), \
                "All elements in prediction must be callable"
        self.activations = activations

    def _compile_loss_weights(self, loss_weights):
        if loss_weights is None:
            loss_weights = [1.] * len(self.outputs)
        else:
            loss_weights = self._compile_args(
                loss_weights, 'loss_weights', default=1.)
        self.loss_weights = loss_weights

    def compile(self,
                optimizer: train.Optimizer,
                metrics=None,
                loss=None,
                activations=None,
                loss_weights=None,
                target_tensors=None,
                **kwargs):
        if optimizer is None:
            raise RuntimeError("An instance of %s must be provided"
                               " to accomplish compiling" % str(
                                train.Optimizer.__class__.__name__))
        self.optimizer = optimizer
        self.loss = loss or []
        self.activations = activations
        self.metrics = metrics or []
        self.loss_weights = loss_weights
        if not self.built:
            logging.info("=>Model was not built, compile will"
                         " delay after first call(after building)")
            return
        logging.info("=>Start compiling......")
        start = time.time()
        self._is_compiled = True
        self.global_step = train.get_or_create_global_step()
        self._compile_activations(self.activations)
        self._compile_loss_functions(self.loss)
        self._compile_loss_weights(self.loss_weights)
        self._compile_target_tensors(target_tensors)
        self._compile_loss()
        self._compile_metrics(self.metrics)
        self._function_kwargs = kwargs
        logging.info("=>Finish compiling in %.4fs" % (time.time() - start))

    def _assert_compiled(self):
        if not self.is_compiled:
            raise RuntimeError("You must compile before using")

    @staticmethod
    def _valid_data(data):
        if isinstance(data, dict):
            raise TypeError("Can not accept with type dict")
        data = to_list(data)
        if not all(isinstance(x, np.ndarray)
                   or F.is_tensor(x) for x in data):
            raise ValueError("All elements should be instances"
                             " of numpy.ndarray or tensorflow.Tensor, but"
                             " received: " + str(data))
        return data

    def build(self, inputs, training=None):
        for i, x in enumerate(inputs):
            name = 'input_%d' % (i + 1)
            self.input_names.append(name)
            if isinstance(x, list):
                x = np.asarray(x)
                if x.ndim == 1:
                    x = np.expand_dims(x, 1)
            if isinstance(x, np.ndarray):
                shape = (None,) + x.shape[1:]
                placeholder = F.placeholder(
                    shape=shape, name=name)
                self.inputs.append(placeholder)
                self._feed_inputs.append(placeholder)
                self._feed_input_names.append(name)
                self._feed_input_shapes.append(shape)
            else:
                self.inputs.append(x)
                if F.is_placeholder(x):
                    self._feed_inputs.append(x)
                    self._feed_input_names.append(name)
                    self._feed_input_shapes.append(F.int_shape(x))
        if has_arg(self.network.forward, 'training'):
            self._uses_learning_phase = True
            self.outputs = to_list(self.network(*self.inputs, training=training))
        else:
            self.outputs = to_list(self.network(*self.inputs))
        if not self.uses_learning_phase:
            self._uses_learning_phase = any(getattr(x, '_uses_learning_phase', False)
                                            for x in self.outputs)
        self.output_names = [
            'output_%d' % i for i in range(1, len(self.outputs) + 1)]
        self._built = True

    def _make_train_function(self, hooks=None):
        self._assert_compiled()
        if self.train_function is None:
            inputs = self._feed_inputs + self._feed_targets
            if self.uses_learning_phase:
                inputs += [F.learning_phase()]
            params = self.network.trainable_weights
            if isgenerator(params):
                params = list(params)
            if len(params) == 0:
                raise ValueError("Parameters can not be empty")
            with ops.name_scope('training'):
                with ops.name_scope(self.optimizer.__class__.__name__):
                    if not hasattr(self.optimizer, 'get_updates'):
                        self.optimizer = Optimizer(
                            self.optimizer, self.global_step)
                    training_updates = self.optimizer.get_updates(
                        params=params, loss=self.total_loss)
                self.train_function = Function(
                    inputs=inputs,
                    outputs=[self.total_loss] + self.metric_tensors,
                    updates=training_updates,
                    hooks=hooks if hooks else [],
                    name='train_function',
                    **self._function_kwargs)

    def _make_test_function(self, hooks=None):
        self._assert_compiled()
        if self.test_function is None:
            inputs = self._feed_inputs + self._feed_targets
            if self.uses_learning_phase:
                inputs += [F.learning_phase()]
            with ops.name_scope('evaluation'):
                self.test_function = Function(
                    inputs=inputs,
                    outputs=[self.total_loss] + self.metric_tensors,
                    hooks=hooks if hooks is not None else [],
                    name='test_function',
                    **self._function_kwargs)

    def _make_predict_function(self):
        self._assert_compiled()
        if self.predict_function is None:
            inputs = self._feed_inputs
            if self.uses_learning_phase:
                inputs += [F.learning_phase()]
            with ops.name_scope('predict'):
                self.predict_function = Function(
                    inputs=inputs,
                    outputs=self.outputs,
                    name='predict_function',
                    **self._function_kwargs)

    def _standardize_data(self,
                          x,
                          y=None):
        x = self._valid_data(x)
        all_inputs = x
        if not self.built:
            self.build(x)
        x = utils.verify_and_normalize_data(
            x,
            self._feed_input_names,
            self._feed_input_shapes)
        if y is not None:
            y = self._valid_data(y)
            all_inputs += y
            if not self.is_compiled:
                self.compile(optimizer=self.optimizer,
                             loss=self.loss,
                             activations=self.activations,
                             metrics=self.metrics,
                             loss_weights=self.loss_weights,
                             target_tensors=y)
            y = utils.verify_and_normalize_data(
                y,
                self._feed_output_names,
                self._feed_output_shapes)
        else:
            y = []
        types = {type(v) for v in all_inputs}
        if len(types) != 1:
            raise ValueError("All elements in x and y should"
                             " have same type, but received:" + str(types))
        elif F.is_tensor(all_inputs[0]):
            x, y = [], []
        return x, y

    def function_loop(self,
                      data,
                      function,
                      sparse_indices=None,
                      batch_size=None,
                      steps=None,
                      shuffle=False,
                      num_samples=None):
        if not sparse_indices:
            sparse_indices = []
        if steps:
            for _ in range(steps):
                function(data)
                if function.should_stop:
                    break
        else:
            batches = utils.make_batches(num_samples, batch_size)
            indices = np.arange(num_samples)
            if shuffle:
                indices = utils.shuffle(indices, batch_size=batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = indices[batch_start: batch_end]
                if self.uses_learning_phase:
                    batch_data = slice_arrays(data[:-1], batch_ids) + [data[-1]]
                else:
                    batch_data = slice_arrays(data, batch_ids)
                for i in sparse_indices:
                    batch_data[i] = batch_data[i].toarray()
                function(batch_data)
                if function.should_stop:
                    break

    def _sparse_data_indices(self, data):
        if not data:
            return []
        feed = self._feed_inputs + self._feed_targets
        sparse_indices = [i for i in range(len(feed))
                          if issparse(data[i]) and
                          not F.is_sparse(feed[i])]
        return sparse_indices

    def _may_restore(self, model_dir):
        if not model_dir:
            return
        if not os.path.exists(model_dir):
            return
        paths = list(list_files(model_dir))
        paths.sort(key=lambda x: os.path.getmtime(x))
        path = '.'.join(os.path.abspath(paths[-1]).split('.')[:2])
        self.network.load_weights(
            path, allow_skip=False, global_step=self.global_step)
        logging.info("=>Restore network: %s from %s"
                     % (self.network.name, path))

    def fit(self,
            x=None,
            y=None,
            val_x=None,
            val_y=None,
            batch_size=None,
            shuffle=True,
            epochs=100,
            hooks=None,
            val_hooks=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None):
        if batch_size is None and steps_per_epoch is None:
            batch_size = 32
        if x is None and y is None and steps_per_epoch is None:
            raise ValueError("When fitting from data tensors,"
                             " `steps_per_epoch` must be specified")
        # build train function
        x, y = self._standardize_data(x, y)
        self._may_restore(self.cfg.get('model_dir', './test_ckpt'))
        data = x + y
        if self.uses_learning_phase:  # [1.] flag for training
            data += [1.]
        hooks = to_list(hooks) if hooks else []
        num_train_samples = utils.check_num_samples(
            data, batch_size=batch_size,
            steps=steps_per_epoch)
        self._make_train_function(hooks=hooks + [
            ProgressHook(target=steps_per_epoch if steps_per_epoch
                         else num_train_samples,
                         epochs=epochs,
                         metric_names=self.metric_names,
                         batch_size=batch_size,
                         stateful_metric_names=self.stateful_metric_names),
            CkPtSaverHook(file_dir=self.cfg.get('model_dir', './test_ckpt'),
                          global_step_tensor=self.global_step,
                          save_steps=self.cfg.get('save_checkpoints_steps', 5))])
        # build val function
        validation = False
        val_data = []
        num_val_samples = None
        if val_x is not None and val_y is not None:
            validation = True
            val_x, val_y = self._standardize_data(val_x, val_y)
            val_data = val_x + val_y
            if self.uses_learning_phase:  # [0.] flag for non-training
                val_data += [0.]
        elif validation_steps:
            if steps_per_epoch is None:
                raise ValueError("When evaluating on train data,"
                                 " `steps_per_epoch` must be set")
            validation = True
            if self._uses_learning_phase:
                val_data = [0.]
        if validation:
            if steps_per_epoch and not validation_steps:
                raise ValueError("When evaluating on train data,"
                                 " `validation_steps` must be set")
            val_hooks = to_list(val_hooks) if val_hooks else []
            num_val_samples = utils.check_num_samples(
                val_data, batch_size=batch_size,
                steps=validation_steps)
            self._make_test_function(hooks=val_hooks + [
                ProgressHook(target=validation_steps if validation_steps
                             else num_val_samples,
                             epochs=epochs,
                             batch_size=batch_size,
                             metric_names=self.metric_names,
                             stateful_metric_names=self.stateful_metric_names)])
        msg = "==>Start training"
        if num_train_samples:
            msg += " on %d samples" % num_train_samples
        if validation:
            msg += ' and evaluating'
            if num_val_samples:
                msg += ' on %d samples' % num_val_samples
        logging.info(msg)
        # To prevent a slowdown, convert sparse array to dense
        sparse_indices = self._sparse_data_indices(data)
        val_sparse_indices = self._sparse_data_indices(val_data)
        for epoch in range(initial_epoch, epochs):
            # Reset Metrics' states
            for metric in self.stateful_metrics:
                metric.reset_states()
            self.function_loop(data,
                               sparse_indices=sparse_indices,
                               function=self.train_function,
                               batch_size=batch_size,
                               steps=steps_per_epoch,
                               shuffle=shuffle,
                               num_samples=num_train_samples)
            if self.train_function.should_stop:
                break
            if validation:
                self.function_loop(val_data,
                                   sparse_indices=val_sparse_indices,
                                   function=self.test_function,
                                   batch_size=batch_size,
                                   steps=validation_steps,
                                   shuffle=False,
                                   num_samples=num_val_samples)
                if self.test_function.should_stop:
                    break
        self.train_function.close()

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 steps=None):
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and y is None and steps is None:
            raise ValueError("If evaluating from data tensors, "
                             "argument `steps` must be set")
        x, y = self._standardize_data(x, y)
        self._may_restore(self.cfg.get('model_dir', './test_ckpt'))
        inputs = x + y
        if self.uses_learning_phase:
            inputs += [0.]
        num_samples = utils.check_num_samples(
            inputs, batch_size=batch_size,
            steps=steps)
        self._make_test_function(
            hooks=[ProgressHook(target=steps if steps else num_samples,
                                epochs=1,
                                batch_size=batch_size,
                                metric_names=self.metric_names,
                                stateful_metric_names=self.stateful_metric_names)])
        sparse_indices = self._sparse_data_indices(inputs)
        self.function_loop(inputs,
                           self.test_function,
                           sparse_indices=sparse_indices,
                           batch_size=batch_size,
                           steps=steps,
                           num_samples=num_samples)
        self.test_function.close()

    def predict(self,
                x,
                batch_size=None,
                steps=None):
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and steps is None:
            raise ValueError('If predicting from data tensors, '
                             'you should specify the `steps` '
                             'argument.')
        x, _ = self._standardize_data(x)
        self._may_restore(self.cfg.get('model_dir', './test_ckpt'))
        inputs = x
        if self.uses_learning_phase:
            inputs += [0.]
        self._make_predict_function()
        sparse_indices = self._sparse_data_indices(inputs)
        num_samples = utils.check_num_samples(
            inputs, batch_size=batch_size,
            steps=steps)
        self.function_loop(inputs,
                           self.predict_function,
                           sparse_indices=sparse_indices,
                           batch_size=batch_size,
                           steps=steps,
                           num_samples=num_samples)
        self.predict_function.close()

    def train_op_batch(self,
                       x,
                       y):
        x, y = self._standardize_data(x, y)
        inputs = x + y
        if self.uses_learning_phase:
            inputs += [1.]
        self._make_train_function()
        outputs = self.train_function(inputs)
        return unpack_singleton(outputs)

    def test_on_batch(self,
                      x,
                      y):
        x, y = self._standardize_data(x, y)
        inputs = x + y
        if self.uses_learning_phase:
            inputs += [0.]
        self._make_test_function()
        outputs = self.test_function(inputs)
        return unpack_singleton(outputs)

    def predict_on_batch(self, x):
        x, _ = self._standardize_data(x)
        inputs = x
        if self.uses_learning_phase:
            inputs += [0.]
        self._make_predict_function()
        outputs = self.predict_function(inputs)
        return unpack_singleton(outputs)
