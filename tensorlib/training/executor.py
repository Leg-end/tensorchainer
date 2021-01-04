import numpy as np
import random
import os
import time
from collections import namedtuple
from scipy.sparse import issparse
from tensorflow import logging
from tensorflow.python import ops
from tensorflow.random import set_random_seed
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops as fops
from tensorlib.engine import base_lib as F
from tensorlib.utils.nest import flatten_list
from tensorlib.utils import to_list, unpack_singleton, slice_arrays
from tensorlib.training import utils
from tensorlib.training.optimizers import Optimizer
from tensorlib.training.sessions.session import Function
from tensorlib.training import ProgressHook, CkPtSaverHook, SummaryHook, PredictHook


class ExecutorMode(object):
    """Standard names for Executor model modes.

          The following standard keys are defined:

          * `TRAIN`: training/fitting mode.
          * `EVAL`: testing/evaluation mode.
          * `PREDICT`: predication/inference mode.
    """

    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'


# noinspection PyArgumentList
class ExecutorSpec(namedtuple('ExecutorSpec', [
        'outputs', 'feed_inputs', 'loss', 'metrics',
        'params', 'train_hooks', 'val_hooks'])):

    def __new__(cls,
                outputs,
                feed_inputs=None,
                loss=None,
                metrics=None,
                params=None,
                train_hooks=None,
                val_hooks=None):
        if outputs is None:
            raise ValueError("Model output must be specified")
        outputs = list(flatten_list(to_list(outputs)))
        if train_hooks is None:
            train_hooks = []
        if val_hooks is None:
            val_hooks = []
        if feed_inputs is not None:
            feed_inputs = to_list(feed_inputs)
        return super(ExecutorSpec, cls).__new__(
            cls, outputs=outputs, feed_inputs=feed_inputs,
            loss=loss, metrics=metrics, params=params,
            train_hooks=train_hooks, val_hooks=val_hooks)


class EnvironmentConfig:
    def __init__(self, **kwargs):
        self.random_seed = 666
        self.CUDA_VISIBLE_DEVICES = '0'
        self.per_process_gpu_memory_fraction = 0.4
        self.intra_op_parallelism_threads = 2
        self.inter_op_parallelism_threads = 8
        self.num_parallel_calls = 4  # tf.dataset.experimental.AUTOTUNE
        self.prefetch_size = 8  # tf.dataset.experimental.AUTOTUNE
        self.allow_growth = True
        self.__dict__.update(**kwargs)

    def __repr__(self):
        value = '\n'.join('\t' + key + ' : ' + str(value)
                          for key, value in self.__dict__.items())
        value += '\n'
        return value


class Executor:

    @property
    def uses_learning_phase(self):
        return self._uses_learning_phase

    @property
    def is_compiled(self):
        return self._is_compiled

    @property
    def built(self):
        return self._built

    @staticmethod
    def _valid_data(data, name='data'):
        values = []
        names = []
        if isinstance(data, dict):
            for name, value in data.items():
                names.append(name)
                values.append(value)
        else:
            values = to_list(data)
            names = [name + '_%d' % i for i in range(1, len(values) + 1)]
        if not all(isinstance(x, np.ndarray)
                   or F.is_tensor(x) for x in values):
            raise ValueError("All elements should be instances"
                             " of numpy.ndarray or tensorflow.Tensor, but"
                             " received: " + str(values))
        return names, values

    @staticmethod
    def _nest_data(names, values, data):
        if isinstance(data, dict):
            data = {name: value for name, value in zip(
                names, values)}
        else:
            data = unpack_singleton(values)
        return data

    @staticmethod
    def _load_global_step(checkpoint_dir):
        try:
            checkpoint_reader = training.NewCheckpointReader(
                training.latest_checkpoint(checkpoint_dir))
            step = checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
            return step
        except Exception as e:
            print("Ignored: " + str(e.args))
            return 0

    def __init__(self, model_fn):
        logging.set_verbosity(logging.INFO)
        assert callable(model_fn)
        self.model_fn = model_fn
        self._mode = None
        self.optimizer = None
        self.loss = None
        self.params = None
        self.metrics = None
        self.outputs = None
        self.train_hooks = []
        self.val_hooks = []
        self.train_function = None
        self.eval_function = None
        self.predict_function = None
        self._predict_hooks = []
        self._checkpoint_dir = None
        self._function_kwargs = {}
        self._session_kwargs = {}
        self._uses_learning_phase = False
        self._is_compiled = False
        self._built = False

    def _compile_environment(self, cfg: EnvironmentConfig):
        logging.info("=>Compiling execution environment...")
        logging.info("=>Execution with environment config:\n %s" % (str(cfg)))
        if self._checkpoint_dir is None:
            raise ValueError("Checkpoint_dir must be specified to "
                             "store execution results")
        # Config random
        random_seed = cfg.random_seed
        np.random.seed(random_seed)
        set_random_seed(random_seed)
        random.seed(random_seed)
        # Config GPU
        if hasattr(cfg, 'CUDA_VISIBLE_DEVICES'):
            logging.info("=>Setting visible devices on %s" % cfg.CUDA_VISIBLE_DEVICES)
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
        intra_op_parallelism_threads = cfg.intra_op_parallelism_threads \
            if hasattr(cfg, 'intra_parallelism_threads') else 0
        inter_op_parallelism_threads = cfg.inter_op_parallelism_threads \
            if hasattr(cfg, 'inter_parallelism_threads') else 0
        session_config = config_pb2.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=inter_op_parallelism_threads,
            intra_op_parallelism_threads=intra_op_parallelism_threads)
        if hasattr(cfg, 'per_process_gpu_memory_fraction'):
            session_config.gpu_options.per_process_gpu_memory_fraction = \
                cfg.per_process_gpu_memory_fraction
        if hasattr(cfg, 'allow_growth'):
            session_config.gpu_options.allow_growth = cfg.allow_growth
        # Create first session
        F.get_session(config=session_config,
                      checkpoint_dir=self._checkpoint_dir)

    def _compile_metrics(self, metrics):
        logging.info("=>Compiling metrics...")
        self.metric_names = ['loss']
        self.metric_tensors = []
        self.stateful_metrics = set()
        self.stateful_metric_names = []
        if isinstance(metrics, dict):
            for name, metric in metrics.items():
                self.metric_names.append(name)
                self.metric_tensors.append(metric)
                if hasattr(metric, '_metric_obj'):
                    self.stateful_metrics.add(getattr(metric, '_metric_obj'))
                    self.stateful_metric_names.append(name)
        else:
            for i, metric in enumerate(to_list(metrics)):
                self.metric_tensors.append(metric)
                name = 'metric_%d' % (i + 1)
                if hasattr(metric, '_metric_obj'):
                    name = getattr(metric, '_metric_obj').name
                    self.stateful_metrics.add(getattr(metric, '_metric_obj'))
                    self.stateful_metric_names.append(name)
                self.metric_names.append(name)

    def _compile_summary(self):
        logging.info("=>Compiling summary...")
        self.summary_ops = fops.get_collection(fops.GraphKeys.SUMMARIES)

    def _compile_args_with_mode(self):
        if self._mode == ExecutorMode.TRAIN:
            if self.optimizer is None:
                raise RuntimeError("An instance of Optimizer must be provided"
                                   " to accomplish compiling")
            assert F.ndim(self.loss) is 0, 'loss must be a scalar tensor(rank is 0),' \
                                           ' but received element %s with rank %s' % (
                                          str(self.loss), str(F.ndim(self.loss)))
            assert len(self.params) > 0, 'params can not be empty'

    def compile(self,
                optimizer=None,
                loss=None,
                metrics=None,
                params=None,
                train_hooks=None,
                val_hooks=None,
                checkpoint_dir=None,
                **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.params = params
        if train_hooks is not None:
            self.train_hooks.extend(to_list(train_hooks))
        if val_hooks is not None:
            self.val_hooks.extend(to_list(val_hooks))
        self._checkpoint_dir = checkpoint_dir
        self._session_kwargs = kwargs
        if not self.built:
            logging.info("=>Model function was not built, fully compile will"
                         " delay after first call(after fit|evaluate|predict)")
            return
        self._is_compiled = True
        self._compile_args_with_mode()
        if self._mode != ExecutorMode.PREDICT:
            self._compile_metrics(metrics)
            self._compile_summary()
        self._compile_environment(EnvironmentConfig(**self._session_kwargs))

    def _prepare_train_hooks(self,
                             epochs,
                             steps_per_epoch,
                             initial_epoch=0):
        saver_hooks = [h for h in self.train_hooks if isinstance(h, CkPtSaverHook)]
        if not saver_hooks:
            self.train_hooks.append(CkPtSaverHook(
                file_dir=self._checkpoint_dir,
                global_step_tensor=training_util.get_global_step(),
                save_steps=steps_per_epoch))
        if self.summary_ops:
            self.train_hooks.append(SummaryHook(
                save_steps=steps_per_epoch,
                summary_op=self.summary_ops,
                output_dir=self._checkpoint_dir + '/train'))
        self.train_hooks.append(ProgressHook(
            title='Training',
            target=steps_per_epoch,
            epochs=epochs,
            initial_epoch=initial_epoch,
            metric_names=self.metric_names,
            stateful_metric_names=self.stateful_metric_names))

    def _prepare_val_hooks(self,
                           epochs,
                           steps_per_epoch,
                           initial_epoch=0):
        if self.summary_ops:
            self.val_hooks.append(SummaryHook(
                save_steps=steps_per_epoch,
                summary_op=self.summary_ops,
                reset_step=True,
                output_dir=self._checkpoint_dir + '/validation'))
        self.val_hooks.append(ProgressHook(
            title='Evaluation',
            target=steps_per_epoch,
            epochs=epochs,
            initial_epoch=initial_epoch,
            metric_names=self.metric_names,
            stateful_metric_names=self.stateful_metric_names))

    def _prepare_predict_hooks(self,
                               steps_per_epoch,
                               predict_fn):
        if self.summary_ops:
            self._predict_hooks.append(SummaryHook(
                save_steps=steps_per_epoch,
                summary_op=self.summary_ops,
                reset_step=True,
                output_dir=self._checkpoint_dir + '/prediction'))
        self._predict_hooks.append(PredictHook(self.inputs, self.outputs, predict_fn))

    def _make_train_function(self):
        self._assert_compiled()
        if self.train_function is None:
            logging.info("=>Creating training function...")
            inputs = self._feed_inputs + self._feed_targets
            if self.uses_learning_phase:
                inputs += [F.learning_phase()]
            with ops.name_scope('training'):
                with ops.name_scope(self.optimizer.__class__.__name__):
                    if not hasattr(self.optimizer, 'get_updates'):
                        self.optimizer = Optimizer(
                            optimizer=self.optimizer,
                            global_step=training_util.get_global_step())
                    # extra updates (e.g. slim.batch_norm)
                    update_ops = fops.get_collection(fops.GraphKeys.UPDATE_OPS)
                    training_updates = self.optimizer.get_updates(
                        params=self.params, loss=self.loss)
                self.train_function = Function(
                    inputs=inputs,
                    outputs=[self.loss] + self.metric_tensors,
                    updates=training_updates + update_ops,
                    name='train_function',
                    hooks=self.train_hooks,
                    **self._function_kwargs)
            logging.info("=>Finish creating training function...")

    def _make_eval_function(self):
        self._assert_compiled()
        if self.eval_function is None:
            logging.info("=>Creating evaluation function...")
            inputs = self._feed_inputs + self._feed_targets
            if self.uses_learning_phase:
                inputs += [F.learning_phase()]
            with ops.name_scope('evaluation'):
                self.eval_function = Function(
                    inputs=inputs,
                    outputs=[self.loss] + self.metric_tensors,
                    name='eval_function',
                    hooks=self.val_hooks,
                    **self._function_kwargs)
            logging.info("=>Finish creating evaluation function...")

    def _make_predict_function(self):
        self._assert_compiled()
        if self.predict_function is None:
            logging.info("=>Creating predict function...")
            inputs = self._feed_inputs
            if self.uses_learning_phase:
                inputs += [F.learning_phase()]
            with ops.name_scope('predict'):
                self.predict_function = Function(
                    inputs=inputs,
                    outputs=self.outputs,
                    hooks=self._predict_hooks,
                    name='predict_function',
                    **self._function_kwargs)
            logging.info("=>Finish creating predict function...")

    def _assert_compiled(self):
        if not self.is_compiled:
            raise RuntimeError("You must compile before using")

    def _build_feed_inputs(self, inputs, names):
        self.inputs = []
        self._input_names = names
        self._feed_inputs = []
        self._feed_input_names = []
        self._feed_input_shapes = []
        for i, x in enumerate(inputs):
            name = names[i]
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

    def _build_feed_targets(self, targets, names):
        self.targets = []
        self._target_names = names
        self._feed_targets = []
        self._feed_target_names = []
        self._feed_target_shapes = []
        for i, x in enumerate(targets):
            name = names[i]
            if isinstance(x, list):
                x = np.asarray(x)
                if x.ndim == 1:
                    x = np.expand_dims(x, 1)
            if isinstance(x, np.ndarray):
                shape = (None,) + x.shape[1:]
                placeholder = F.placeholder(
                    shape=shape, name=name)
                self.targets.append(placeholder)
                self._feed_targets.append(placeholder)
                self._feed_target_names.append(name)
                self._feed_target_shapes.append(shape)
            else:
                self.targets.append(x)
                if F.is_placeholder(x):
                    self._feed_targets.append(x)
                    self._feed_target_names.append(name)
                    self._feed_target_shapes.append(F.int_shape(x))

    def _build_model_fn(self, x, y=None):
        all_inputs = []
        if not self.built:
            logging.info("=>Building feed inputs and targets...")
            names, inputs = self._valid_data(x, 'inputs')
            all_inputs += inputs
            self._build_feed_inputs(inputs, names)
            self._built = y is None
            if not self.built:
                names, targets = self._valid_data(y, 'targets')
                all_inputs += targets
                self._build_feed_targets(targets, names)
                self._built = True
        else:
            all_inputs += self._valid_data(x, 'inputs')[1]
            if y is not None:
                all_inputs += self._valid_data(y, 'targets')[1]
        types = {type(v) for v in all_inputs}
        if len(types) > 1:
            raise ValueError("All elements in x and y should"
                             " have same type, but received:" + str(types))
        if not self.is_compiled:
            inputs = self._nest_data(self._input_names, self.inputs, x)
            if y is not None:
                targets = self._nest_data(self._target_names, self.targets, y)
            else:
                targets = None
            start = time.time()
            logging.info('=>Calling model_fn...')
            result = self.model_fn(inputs, targets)
            logging.info('=>Finish calling model_fn...')
            if not isinstance(result, ExecutorSpec):
                raise ValueError("Result returned from `model_fn` must be"
                                 "an instance of `ExecutorSpec`")
            self.compile(optimizer=self.optimizer,
                         loss=result.loss,
                         metrics=result.metrics,
                         params=result.params,
                         train_hooks=result.train_hooks,
                         val_hooks=result.val_hooks,
                         checkpoint_dir=self._checkpoint_dir,
                         **self._session_kwargs)
            logging.info("=>Finish compiling in %.4fs" % (time.time() - start))
            # For topological graph model
            if result.feed_inputs is not None:
                feed_names = [placeholder.name for placeholder in result.feed_inputs]
                feed_shapes = [placeholder.shape for placeholder in result.feed_inputs]
                self.inputs = result.feed_inputs + self.inputs
                self._input_names = feed_names + self._input_names
                self._feed_inputs = result.feed_inputs + self._feed_inputs
                self._feed_input_names = feed_names + self._feed_input_names
                self._feed_input_shapes = feed_shapes + self._feed_input_shapes
            self.outputs = result.outputs
            self._uses_learning_phase = hasattr(
                self.outputs[0], '_uses_learning_phase')
        return all_inputs

    def _standardize_data(self,
                          x,
                          y=None):
        """
        This procedure transform any elements in x and y that are not
        placeholder to placeholder
        """
        all_inputs = self._build_model_fn(x, y)
        # If `x` and `y` were all symbolic,
        # then the model should not be fed any inputs and targets.
        # Note: in this case, `any` and `all` are equivalent since we disallow
        # mixed symbolic/value inputs.
        if any(F.is_tensor(v) for v in all_inputs):
            return [], []
        # What follows is input validation and standardization to list format,
        # in the case where all inputs are value arrays.
        x = utils.verify_and_normalize_data(
            x,
            self._feed_input_names,
            self._feed_input_shapes)
        if y is not None:
            y = utils.verify_and_normalize_data(
                y,
                self._feed_target_names,
                self._feed_target_shapes)
            utils.check_array_length_consistency(x, y)
        else:
            y = []
        return x, y

    def _sparse_data_indices(self, data):
        if not data:
            return []
        feed = self._feed_inputs + self._feed_targets
        sparse_indices = [i for i in range(len(feed))
                          if issparse(data[i]) and
                          not F.is_sparse(feed[i])]
        return sparse_indices

    def function_loop(self,
                      data,
                      function,
                      sparse_indices=None,
                      batch_size=None,
                      steps=None,
                      shuffle=False,
                      num_samples=None):
        if steps:
            for _ in range(steps):
                function(data)
                if function.should_stop:
                    break
        else:
            if not sparse_indices:
                sparse_indices = []
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

    def fit_loop(self,
                 data,
                 val_data=None,
                 batch_size=None,
                 shuffle=True,
                 epochs=1,
                 initial_epoch=0,
                 num_train_samples=None,
                 num_val_samples=None,
                 steps_per_epoch=None,
                 validation_steps=None):
        validation = self.eval_function and val_data
        if validation_steps:
            validation = True
            if steps_per_epoch is None:
                raise ValueError('Can only use `validation_steps` '
                                 'when doing step-wise '
                                 'training, i.e. `steps_per_epoch` '
                                 'must be set.')
        elif validation:
            if steps_per_epoch:
                raise ValueError('Must specify `validation_steps` '
                                 'to perform validation '
                                 'when doing step-wise training.')
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
                                   function=self.eval_function,
                                   batch_size=batch_size,
                                   steps=validation_steps,
                                   shuffle=False,
                                   num_samples=num_val_samples)
                if self.eval_function.should_stop:
                    break
        self.train_function.end()
        self.eval_function.end()

    def fit(self,
            x=None,
            y=None,
            val_x=None,
            val_y=None,
            batch_size=None,
            shuffle=True,
            epochs=1,
            steps_per_epoch=None,
            validation_steps=None):
        if batch_size is None and steps_per_epoch is None:
            batch_size = 32
        if x is None and y is None and steps_per_epoch is None:
            raise ValueError("When fitting from data tensors,"
                             " `steps_per_epoch` must be specified")
        self._mode = ExecutorMode.TRAIN
        # prepare global_step
        step = self._load_global_step(self._checkpoint_dir)
        if training_util.get_global_step() is None:
            fops.add_to_collection(
                name=fops.GraphKeys.GLOBAL_STEP,
                value=variables.Variable(
                    step, name='global_step', trainable=False))
        # build train function
        x, y = self._standardize_data(x, y)
        data = x + y
        if self.uses_learning_phase:  # [1.] flag for training
            data += [1.]
        num_train_samples = utils.check_num_samples(
            data, batch_size=batch_size,
            steps=steps_per_epoch)
        train_steps = steps_per_epoch or (
                (num_train_samples + batch_size - 1) // batch_size)
        initial_epoch = step // train_steps
        if epochs is not None:
            if epochs <= initial_epoch:
                logging.info("=>Skipping training since max epoch has already arrived")
                exit(0)
        self._prepare_train_hooks(epochs=epochs,
                                  steps_per_epoch=train_steps,
                                  initial_epoch=initial_epoch)
        self._make_train_function()
        # build val function
        validation = False
        num_val_samples = None
        if val_x is not None and val_y is not None:
            validation = True
            val_x, val_y = self._standardize_data(val_x, val_y)
            val_data = val_x + val_y
        elif validation_steps:
            validation = True
            val_data = []
        else:
            val_data = []
        if validation:
            if self.uses_learning_phase:  # [0.] flag for evaluation
                val_data += [0.]
            num_val_samples = utils.check_num_samples(
                val_data, batch_size=batch_size,
                steps=validation_steps)
            val_steps = validation_steps or (
                    (num_val_samples + batch_size - 1) // batch_size)
            self._prepare_val_hooks(epochs=epochs,
                                    steps_per_epoch=val_steps,
                                    initial_epoch=initial_epoch)
            self._make_eval_function()
        self.fit_loop(data=data,
                      val_data=val_data,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      epochs=epochs,
                      initial_epoch=initial_epoch,
                      steps_per_epoch=steps_per_epoch,
                      validation_steps=validation_steps,
                      num_train_samples=num_train_samples,
                      num_val_samples=num_val_samples)

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 validation_steps=None):
        if batch_size is None and validation_steps is None:
            batch_size = 32
        if x is None and y is None and validation_steps is None:
            raise ValueError("If evaluating from data tensors, "
                             "argument `validation_steps` must be set")
        self._mode = ExecutorMode.EVAL
        x, y = self._standardize_data(x, y)
        inputs = x + y
        if self.uses_learning_phase:
            inputs += [0.]
        num_samples = utils.check_num_samples(
            inputs, batch_size=batch_size,
            steps=validation_steps)
        steps = validation_steps or (
                (num_samples + batch_size - 1) // batch_size)
        self._prepare_val_hooks(epochs=1,
                                steps_per_epoch=steps,
                                initial_epoch=0)
        self._make_eval_function()
        msg = "==>Start evaluating"
        if num_samples:
            msg += " on %d samples" % num_samples
        logging.info(msg)
        sparse_indices = self._sparse_data_indices(inputs)
        self.function_loop(inputs,
                           self.eval_function,
                           sparse_indices=sparse_indices,
                           batch_size=batch_size,
                           steps=validation_steps,
                           num_samples=num_samples)
        self.eval_function.end()

    def predict(self,
                x,
                batch_size=None,
                steps=None,
                predict_fn=None):
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and steps is None:
            raise ValueError('If predicting from data tensors, '
                             'you should specify the `steps` '
                             'argument.')
        self._mode = ExecutorMode.PREDICT
        x, _ = self._standardize_data(x)
        inputs = x
        if self.uses_learning_phase:
            inputs += [0.]
        num_samples = utils.check_num_samples(
            inputs, batch_size=batch_size,
            steps=steps)
        steps = steps or ((num_samples + batch_size - 1) // batch_size)
        self._prepare_predict_hooks(steps_per_epoch=steps,
                                    predict_fn=predict_fn)
        self._make_predict_function()
        msg = "==>Start predicting"
        if num_samples:
            msg += " on %d samples" % num_samples
        logging.info(msg)
        sparse_indices = self._sparse_data_indices(inputs)
        self.function_loop(inputs,
                           self.predict_function,
                           sparse_indices=sparse_indices,
                           batch_size=batch_size,
                           steps=steps,
                           num_samples=num_samples)
        self.predict_function.end()

    def train_op_batch(self,
                       x,
                       y):
        self._mode = ExecutorMode.TRAIN
        x, y = self._standardize_data(x, y)
        inputs = x + y
        if self.uses_learning_phase:
            inputs += [1.]
        self._make_train_function()
        outputs = self.train_function(inputs)
        return unpack_singleton(outputs)

    def eval_on_batch(self,
                      x,
                      y):
        self._mode = ExecutorMode.EVAL
        x, y = self._standardize_data(x, y)
        inputs = x + y
        if self.uses_learning_phase:
            inputs += [0.]
        self._make_eval_function()
        outputs = self.eval_function(inputs)
        return unpack_singleton(outputs)

    def predict_on_batch(self, x):
        self._mode = ExecutorMode.PREDICT
        x, _ = self._standardize_data(x)
        inputs = x
        if self.uses_learning_phase:
            inputs += [0.]
        self._make_predict_function()
        outputs = self.predict_function(inputs)
        return unpack_singleton(outputs)
