import numpy as np
import random
import os
import time
import toml
from collections import namedtuple
from scipy.sparse import issparse
from tensorflow import logging
from tensorflow.python import ops
from tensorflow.random import set_random_seed
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training import training as tf_training
from tensorflow.python.training import training_util
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops as fops
from tensorlib.engine import base_lib as F
from tensorlib.engine import Network
from tensorlib.utils import nest
from tensorlib.utils import to_list, unpack_singleton, slice_arrays, has_arg
from tensorlib.training import utils
from tensorlib.training import losses
from tensorlib.training import metrics as metric_module
from tensorlib.training.optimizers import Optimizer
from tensorlib.training.sessions.session import Function
from tensorlib.training import ProgressHook, CkPtSaverHook, SummaryHook, PredictHook


# noinspection PyArgumentList
class EstimatorSpec(namedtuple('ExecutorSpec', [
        'outputs', 'loss', 'metrics',
        'train_hooks', 'val_hooks'])):

    def __new__(cls,
                outputs=None,
                loss=None,
                metrics=None,
                train_hooks=None,
                val_hooks=None):
        if outputs is not None:
            outputs = nest.flatten(to_list(outputs))
        if train_hooks is None:
            train_hooks = []
        if val_hooks is None:
            val_hooks = []
        return super(EstimatorSpec, cls).__new__(
            cls, outputs=outputs, loss=loss,
            metrics=metrics, train_hooks=train_hooks,
            val_hooks=val_hooks)


class Model(Network):
    """
    Group of layers with training, evaluating and inference features

    # Instantiation

    1. Create Graph-Model
        >> import tensorlib as lib
    
        >> inputs = lib.Input(shape=(3,))
        >> x = lib.layers.Dense(units=4)(inputs)
        >> outputs = lib.layers.Dense(units=5)(x)
        >> model = lib.training.Model(inputs=inputs, outputs=outputs)
    2. Create subclassed Model
        >> import tensorlib as lib
        
        >> class MyModel(lib.Model):
            
            def __init__(self):
                super(MyModel, self).__init__()
                self.dense1 = lib.layers.Dense(units=4)
                self.dense2 = lib.layers.Dense(units=5)
                
            def forward(self, inputs):
                x = self.dense1(inputs)
                return self.dense2(x)
                
        >> model = MyModel()
        If you need this subclassed model has dynamic learning phase, you
        can alter "def forward(self, inputs)" to "def forward(self, inputs, training=False)"

    # Execution usages

    We support 2 mechanisms for doing execution stuff After creating model
    1. Keras-style 
        >> import tensorflow as tf
        
        >> model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                         loss='mse',
                         metrics='mse',
                         checkpoint_dir='./checkpoint')

        # Use numpy data

        >> model.fit(x=train_x, y=train_y, epochs=100)

        # Use tf.data.Dataset

        >> model.fit(x=train_x, y=train_y, epochs=100, steps_per_epoch=1000)
    2. estimator-style
        >> import tensorflow as tf
        
        >> def model_fn(model, inputs, labels):
               # do data preprocess....
               # prepare customize hooks
               outputs = model(inputs)
               # calculate loss and metrics...
               return lib.training.EstimatorSpec(outputs=outputs,
                                                train_hooks=...,
                                                val_hooks=...,
                                                loss=...,
                                                metrics=...)
        >> model.compile(model_fn=model_fn,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                         checkpoint_dir='./checkpoint')
        # Use numpy data

        >> model.fit(x=train_x, y=train_y, epochs=100)

        # Use tf.data.Dataset

        >> model.fit(x=train_x, y=train_y, epochs=100, steps_per_epoch=1000)

    # Summary of execution implementation

    1. Standardize data
        Turn data to placeholder if it is numpy format, and gather
        all data placeholder
    2. Build model
        Generate computation graph of model by calling model or model_fn
    3. Compile model for execution
        Compile all graph node (e.g. loss, metrics...) associated with data placeholder
    4. Prepare execution function to execute computation graph
        Map calculation needed node (e.g. loss, metrics, updates) to data placeholder
        by lib.base_lib.Function and calculate them step-wise while epoch-wise
    5. Handle computing results by lib.training.SessRunHook
    """

    @property
    def uses_learning_phase(self):
        return self._uses_learning_phase

    @property
    def is_compiled(self):
        return self._is_compiled

    @property
    def is_built(self):
        return self._is_built

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        logging.set_verbosity(logging.INFO)
        
        self.model_fn = None
        self.optimizer = None
        self.loss = None
        self.loss_weights = None
        self.metrics = None

        self._input_names = []
        self._feed_inputs = []
        self._feed_input_names = []
        self._feed_input_shapes = []
        self._output_names = []

        self.train_hooks = []
        self.val_hooks = []
        self._predict_hooks = []

        self.train_function = None
        self.eval_function = None
        self.predict_function = None

        self._checkpoint_dir = None
        self._function_kwargs = {}
        self._session_cfg = None
        self._uses_learning_phase = False
        self._is_compiled = False
        self._is_built = False

    def _load_global_step(self):
        try:
            checkpoint_reader = tf_training.NewCheckpointReader(
                tf_training.latest_checkpoint(self._checkpoint_dir))
            step = checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
            return step
        except Exception as e:
            print("Ignored: " + str(e.args))
            return 0

    def _compile_environment(self, _session_cfg):
        cfg = toml.load('./env_cfg.toml')['ENV']
        if isinstance(_session_cfg, str):
            suffix = _session_cfg.split('.')[-1]
            if suffix == 'toml':
                cfg.update(**toml.load(_session_cfg)['ENV'])
            elif suffix == 'json':
                import json
                cfg.update(**json.load(open(_session_cfg)))
            else:
                raise ValueError("Unsupported file format %s, only support toml, json" % suffix)
        elif isinstance(_session_cfg, dict):
            cfg.update(**_session_cfg)
        logging.info("=>Compiling execution environment...")
        logging.info("=>Execution with environment config:\n %s" % (str(cfg)))
        if self._checkpoint_dir is None:
            raise ValueError("Checkpoint_dir must be specified to "
                             "store execution results")
        # Config random
        random_seed = cfg['random_seed']
        np.random.seed(random_seed)
        set_random_seed(random_seed)
        random.seed(random_seed)
        # Config GPU
        if hasattr(cfg, 'CUDA_VISIBLE_DEVICES'):
            logging.info("=>Setting visible devices on %s" % cfg['CUDA_VISIBLE_DEVICES'])
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']
        intra_op_parallelism_threads = cfg['intra_op_parallelism_threads'] \
            if hasattr(cfg, 'intra_parallelism_threads') else 0
        inter_op_parallelism_threads = cfg['inter_op_parallelism_threads'] \
            if hasattr(cfg, 'inter_parallelism_threads') else 0
        session_config = config_pb2.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=inter_op_parallelism_threads,
            intra_op_parallelism_threads=intra_op_parallelism_threads)
        if hasattr(cfg, 'per_process_gpu_memory_fraction'):
            session_config.gpu_options.per_process_gpu_memory_fraction = \
                cfg['per_process_gpu_memory_fraction']
        if hasattr(cfg, 'allow_growth'):
            session_config.gpu_options.allow_growth = cfg['allow_growth']
        # Create first session
        F.get_session(config=session_config,
                      checkpoint_dir=self._checkpoint_dir)

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

    def _compile_loss_function(self, loss):
        loss = self._compile_args(loss, 'loss')
        self.loss_functions = [losses.get(name) for name in loss]
        self._skip_target_indices = [i for i, fn in enumerate(
            self.loss_functions) if fn is None]

    def _compile_loss_weights(self, loss_weights):
        if loss_weights is None:
            loss_weights = [1.] * len(self.outputs)
        else:
            loss_weights = self._compile_args(
                loss_weights, 'loss_weights', default=1.)
        self.loss_weights = loss_weights

    def _compile_targets(self, targets):
        logging.info("=>Compiling targets...")
        self.targets = []
        self._feed_targets = []
        self._feed_target_names = []
        self._feed_target_shapes = []
        self._feed_loss_fns = []
        targets = self._compile_args(targets, 'targets')
        for i in range(len(self.outputs)):
            if i in self._skip_target_indices:
                self.targets.append(None)
            else:
                name = self.output_names[i]
                output = self.outputs[i]
                target = targets[i]
                loss_fn = self.loss_functions[i]
                if target is None:
                    target = F.placeholder(
                        ndim=len(F.int_shape(output)),
                        name=name + '_target',
                        sparse=F.is_sparse(output),
                        dtype=F.dtype(output))
                elif isinstance(target, list):
                    target = np.asarray(target)
                    if target.ndim == 1:
                        target = np.expand_dims(target, 1)
                if isinstance(target, np.ndarray):
                    shape = (None,) + target.shape[1:]
                    placeholder = F.placeholder(
                        shape=shape, name=name)
                    self.targets.append(placeholder)
                    self._feed_targets.append(placeholder)
                    self._feed_target_names.append(name)
                    self._feed_target_shapes.append(shape)
                    self._feed_loss_fns.append(loss_fn)
                else:
                    self.targets.append(target)
                    if F.is_placeholder(target):
                        self._feed_targets.append(target)
                        self._feed_target_names.append(name)
                        self._feed_target_shapes.append(F.int_shape(target))
                        self._feed_loss_fns.append(loss_fn)

    def _compile_loss(self, loss, loss_weights, targets):
        logging.info("=>Compiling loss...")
        self.metric_names = ['loss']  # map with total_loss
        self.metric_tensors = []
        with ops.name_scope('compile_loss'):
            if targets is not None:  # else loss has already been a tensor
                total_loss = 0.
                self._compile_loss_function(loss)
                self._compile_loss_weights(loss_weights)
                self._compile_targets(targets)
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
                loss = total_loss
            reg_loss = fops.get_collection(fops.GraphKeys.REGULARIZATION_LOSSES)
            if reg_loss:
                loss = math_ops.add_n(reg_loss + [loss])
        self.loss = loss

    def _compile_metric_tensors(self, metrics):
        self.stateful_metrics = set()
        self.stateful_metric_names = []
        for name, metric in metrics:
            self.metric_tensors.append(metric)
            self.metric_names.append(name)
            if hasattr(metric, '_metric_obj'):
                self.stateful_metrics.add(getattr(metric, '_metric_obj'))
                self.stateful_metric_names.append(name)

    def _compile_metrics(self, metrics):
        """
        Compile metrics to desired format
            each output map with a list of metrics
            item inside metrics can be an instance of `training.Metric` or a tensor
        Note:
            when metrics if class-format, we will do formation check between metrics
            and `self.outputs` to make sure enough number of metrics to compatible with
            `self.outputs` and `self.targets`
            when metrics if tensor-format, we will not do formation check, cause metric
            calculation already handled by users themselves inside `model_fn`
        :param metrics: None or a nested list or dict
        """
        logging.info("=>Compiling metrics...")
        is_tensor = False
        if not metrics:
            metrics = [[]] * len(self.outputs)
        elif isinstance(metrics, list):
            if not F.is_tensor(metrics[0]):
                if not is_tensor and len(metrics) != len(self.outputs):
                    raise ValueError("Number of metric inside `metrics`"
                                     " %d is not compatible with number"
                                     " of `self.outputs` %d" % (
                                         len(metrics), len(self.outputs)))
            else:
                is_tensor = True
                metrics = [('metric_%d' % (i+1), m) for i, m in enumerate(metrics)]
        elif isinstance(metrics, dict):
            if not F.is_tensor(metrics[list(metrics.keys())[0]]):
                metrics = [metrics.get(name, [])
                           for name in self.output_names]
            else:
                is_tensor = True
                metrics = list(metrics.items())
        else:
            raise TypeError("Unexpected type of metrics: " + str(type(metrics)))
        with ops.name_scope('compile_metric'):
            if is_tensor:
                self._compile_metric_tensors(metrics)
            else:
                # Must handle sparse situation carefully!
                def _compile_metric(m, loss_fn):
                    if isinstance(loss_fn, losses.SparseCategoricalCrossEntropy):
                        if m in {'accuracy', 'acc'}:
                            m = metric_module.SparseCategoricalAccuracy()
                            return m
                    m = metric_module.get(m)
                    return m
                metric_tensors = []
                for i in range(len(self.outputs)):
                    if i in self._skip_target_indices:
                        continue
                    target = self.targets[i]
                    output = self.outputs[i]
                    output_metrics = to_list(metrics[i])
                    loss_function = self.loss_functions[i]
                    for j, metric in enumerate(output_metrics):
                        metric = _compile_metric(metric, loss_function)
                        metric_name = getattr(metric, 'name', 'metric_%d' % j)
                        metric_result = metric(target, output)
                        if len(self.output_names) > 1:
                            metric_name = self.output_names[i] + '_' + metric_name
                        metric_tensors.append((metric_name, metric_result))
                self._compile_metric_tensors(metric_tensors)

    def _compile_summary(self):
        logging.info("=>Compiling summary...")
        self.summary_ops = fops.get_collection(fops.GraphKeys.SUMMARIES)

    def compile(self,
                model_fn=None,
                optimizer=None,
                loss=None,
                loss_weights=None,
                metrics=None,
                train_hooks=None,
                val_hooks=None,
                checkpoint_dir=None,
                targets=None,
                session_cfg=None,
                **kwargs):
        """
        :param model_fn: Function implemented by user when using estimator-style
            execution mechanism, format as:
            Params:
                model: Instance reference of this model, you must call this model
                    to generate computation graph
                inputs: list or dict, input data
                labels: list or dict, labels
            return: lib.training.EstimatorSpec
            def model_fn(model, inputs, labels):
                # data preprocess, hook preparation...
                outputs = model(inputs)
                # calculate loss, metrics, ....
                return lib.training.EstimatorSpec(....)
        :param optimizer: An instance of tf.train.Optimizer
        :param loss: An instance of lib.training.Loss or predefined name of
            lib.training.Loss (e.g. mse) when in keras-style execution mechanism,
             otherwise, tensor computed from model_fn
        :param loss_weights: Optional list or dict specifying scalar
            coefficients (Python floats) to weight the loss contributions
            of different model outputs.
        :param metrics: Nested list with compatible instances of lib.training.Metric
            or predefined name of lib.training.Metric (e.g. [['acc', 'mse'], ['acc']])
            to self.outputs when in keras-style execution mechanism, otherwise, list or
            dict with item as tensor computed from model_fn
        :param train_hooks: List of instances of lib.training.SessRunHook for training,
            it can be passed from model_fn
        :param val_hooks: List of instances of lib.training.SessRunHook for evaluating,
            it can be passed from model_fn
        :param checkpoint_dir: Directory where execution results store in
        :param targets: List or dict target data when in keras-style execution mechanism,
            otherwise, None
        :param session_cfg: Dict or file path contains session config should match content in './env_cfg.toml'
        :param kwargs: Optional function parameters
        """
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.metrics = metrics
        if train_hooks is not None:
            self.train_hooks.extend(to_list(train_hooks))
        if val_hooks is not None:
            self.val_hooks.extend(to_list(val_hooks))
        self._checkpoint_dir = checkpoint_dir
        self._session_cfg = session_cfg
        self._function_kwargs = kwargs
        if not self.is_built:
            logging.info("=>Model function was not built, fully compile will"
                         " delay after first call(after fit|evaluate|predict)")
            return
        start = time.time()
        logging.info("=>Start compiling......")
        self._is_compiled = True
        self._compile_loss(loss=loss,
                           loss_weights=loss_weights,
                           targets=targets)
        self._compile_metrics(metrics)
        self._compile_summary()
        self._compile_environment(self._session_cfg)
        logging.info("=>Finish compiling in %.4fs" % (time.time() - start))

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
                        params=list(self.trainable_weights), loss=self.loss)
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

    def _build_feed_inputs(self, inputs):
        self._input_names = []
        self._feed_inputs = []
        self._feed_input_names = []
        self._feed_input_shapes = []
        self.inputs = []
        for i, x in enumerate(inputs):
            name = 'input_%d' % (i + 1)
            self._input_names.append(name)
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

    def _build_feed_targets(self, targets):
        # We don't check targets' length to compatible with self.outputs'
        # cause loss and metric have already calculated from model_fn
        self.targets = []
        self._target_names = []
        self._feed_targets = []
        self._feed_target_names = []
        self._feed_target_shapes = []
        for i, x in enumerate(targets):
            name = 'target_%d' % (i + 1)
            self._target_names.append(name)
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

    def _set_inputs(self, inputs, outputs=None, training=None):
        """
        Subclassed model
        :param inputs: Only support nested list, non-nested dict;
        :param outputs:
        :param training:
        :return:
        """
        self._nested_inputs = inputs
        self.inputs = []
        for i, x in enumerate(utils.valid_data(inputs)):
            name = 'input_%d' % (i + 1)
            self._input_names.append(name)
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
        if self.model_fn is None:
            kwargs = {'training': training} if has_arg(self.forward, 'training') else {}
            self._nested_outputs = self(inputs, **kwargs)
            self.outputs = nest.flatten(self._nested_outputs)
        elif outputs is not None:
            logging.info('=>Calling model_fn...')
            result = self.model_fn(
                self, utils.nest_data(
                    self.inputs, x_keys, x),
                utils.nest_data(
                    self.targets, y_keys, y))
            logging.info('=>Finish calling model_fn...')
            if not isinstance(result, EstimatorSpec):
                raise ValueError("Result returned from `model_fn` must be"
                                 "an instance of `EstimatorSpec`")
            self.train_hooks.extend(result.train_hooks)
            self.val_hooks.extend(result.val_hooks)
            self.loss = result.loss
            self.metrics = result.metrics
            self.outputs = result.outputs
        self._output_names = [
            'output_%d' % i for i in range(1, len(self.outputs) + 1)]
        self._uses_learning_phase = any(getattr(x, '_uses_learning_phase', False)
                                        for x in self.outputs)
        self.built = True

    def build_model(self, x, y=None, training=None):
        x_keys, valid_x = utils.valid_data(x)
        y_keys, valid_y = utils.valid_data(y)  # y is [] if y=None
        if self.inputs is None:
            self._build_feed_inputs(valid_x)
            if self.model_fn is None:
                if has_arg(self.forward, 'training'):
                    self._uses_learning_phase = True
                    self.outputs = to_list(self(*self.inputs, training=training))
                else:
                    self.outputs = to_list(self(*self.inputs))
            elif y is not None:
                self._build_feed_targets(valid_y)
                logging.info('=>Calling model_fn...')
                result = self.model_fn(
                    self, utils.nest_data(
                        self.inputs, x_keys, x),
                    utils.nest_data(
                        self.targets, y_keys, y))
                logging.info('=>Finish calling model_fn...')
                if not isinstance(result, EstimatorSpec):
                    raise ValueError("Result returned from `model_fn` must be"
                                     "an instance of `EstimatorSpec`")
                self.train_hooks.extend(result.train_hooks)
                self.val_hooks.extend(result.val_hooks)
                self.loss = result.loss
                self.metrics = result.metrics
                self.outputs = result.outputs
        else:  # graph-model, inputs and outputs already satisfied
            self._input_names = []
            self._feed_inputs = []
            self._feed_input_names = []
            self._feed_input_shapes = []
            for i, x in enumerate(self.inputs):
                name = 'input_%d' % (i + 1)
                self._input_names.append(name)
                self._feed_inputs.append(x)
                self._feed_input_names.append(name)
                self._feed_input_shapes.append(F.int_shape(x))
            if self.model_fn is not None:
                self._build_feed_targets(valid_y)
                logging.info('=>Calling model_fn...')
                result = self.model_fn(
                    self, None, utils.nest_data(
                        self.targets, y_keys, y))
                logging.info('=>Finish calling model_fn...')
                if not isinstance(result, EstimatorSpec):
                    raise ValueError("Result returned from `model_fn` must be"
                                     "an instance of `EstimatorSpec`")
                self.train_hooks.extend(result.train_hooks)
                self.val_hooks.extend(result.val_hooks)
                self.loss = result.loss
                self.metrics = result.metrics
                self.outputs = result.outputs
        self._output_names = [
            'output_%d' % i for i in range(1, len(self.outputs) + 1)]
        if not self.uses_learning_phase:
            self._uses_learning_phase = any(getattr(x, '_uses_learning_phase', False)
                                            for x in self.outputs)
        self._is_built = True
        return valid_x, valid_y

    def _standardize_data(self,
                          x,
                          y=None):
        """
        This procedure transform any elements in x and y that are not
        placeholder to placeholder
        """
        # Build the model using the retrieved inputs (value or symbolic).
        # If values, then in symbolic-mode placeholders will be created
        # to match the value shapes.
        if not self.is_built:
            x, y = self.build_model(x, y=y)  # y is [] if y=None
        else:
            _, x = utils.valid_data(x)
            _, y = utils.valid_data(y)  # y is [] if y=None
        if y is not None and y is not [] and not self.is_compiled:
            self.compile(optimizer=self.optimizer,
                         loss=self.loss,
                         loss_weights=self.loss_weights,
                         metrics=self.metrics,
                         checkpoint_dir=self._checkpoint_dir,
                         targets=None if self.model_fn else y,
                         session_cfg=self._session_cfg,
                         **self._function_kwargs)
        # If `x` and `y` were all symbolic,
        # then the model should not be fed any inputs and targets.
        # Note: in this case, `any` and `all` are equivalent since we disallow
        # mixed symbolic/value inputs.
        if any(F.is_tensor(v) for v in x + y):
            return [], []
        # What follows is input validation and standardization to list format,
        # in the case where all inputs are value arrays.
        x = utils.verify_and_normalize_data(
            x,
            self._feed_input_names,
            self._feed_input_shapes)
        if y is not None and y is not []:
            y = utils.verify_and_normalize_data(
                y,
                self._feed_target_names,
                self._feed_target_shapes)
            utils.check_array_length_consistency(x, y)
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
        # prepare global_step
        step = self._load_global_step()
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
                predict_fn,
                batch_size=None,
                steps=None):
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and steps is None:
            raise ValueError('If predicting from data tensors, '
                             'you should specify the `steps` '
                             'argument.')
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
        x, y = self._standardize_data(x, y)
        inputs = x + y
        if self.uses_learning_phase:
            inputs += [0.]
        self._make_eval_function()
        outputs = self.eval_function(inputs)
        return unpack_singleton(outputs)

    def predict_on_batch(self, x):
        x, _ = self._standardize_data(x)
        inputs = x
        if self.uses_learning_phase:
            inputs += [0.]
        self._make_predict_function()
        outputs = self.predict_function(inputs)
        return unpack_singleton(outputs)
