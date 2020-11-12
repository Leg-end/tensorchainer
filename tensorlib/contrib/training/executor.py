from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.estimator import estimator_lib as estimator
import time
from datetime import datetime
import random
import numpy as np
from tensorlib.contrib.training.hparams import *
import os
from tensorlib import saving
import tensorflow as tf


# os.environ[“CUDA_DEVICE_ORDER”] = “PCI_BUS_ID” # 按照PCI_BUS_ID顺序从0开始排列GPU设备
# os.environ[“CUDA_VISIBLE_DEVICES”] = “0” #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
# os.environ[“CUDA_VISIBLE_DEVICES”] = “1” #设置当前使用的GPU设备仅为1号设备  设备名称为'/gpu:0'
# os.environ[“CUDA_VISIBLE_DEVICES”] = “0,1” #设置当前使用的GPU设备为0,1号两个设备,名称依次为'/gpu:0'、'/gpu:1'
# os.environ[“CUDA_VISIBLE_DEVICES”] = “1,0” #设置当前使用的GPU设备为1,0号两个设备,名称依次为'/gpu:0'、'/gpu:1'。表示优先使用1号设备,然后使用0号设备


class LoggerHook(tf.train.SessionRunHook):
    def __init__(self, learning_rate, log_frequency,
                 batch_size, loss=None, accuracy=None,
                 metric_names=None):
        self._loss = loss
        self._acc = accuracy
        self._lr = learning_rate
        self._log_freq = log_frequency
        self._batch_size = batch_size
        self._step = None
        self._start_time = None
        if isinstance(accuracy, (list, tuple)):
            self.metric_names = metric_names or ['{:d}th: '.format(i) for i in range(len(accuracy))]

    def begin(self):
        if self._loss is None:
            self._loss = tf.get_collection(tf.GraphKeys.LOSSES)
        if self._acc is None:
            self._acc = tf.constant(-1.)
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([self._loss, self._lr, self._acc])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        if self._step % self._log_freq == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss, lr, acc = run_values.results
            if hasattr(acc, '__len__'):
                acc_str = ' | '.join(['{}: {:.3f}'.format(name, value)
                                      for name, value in zip(self.metric_names, acc)])
            else:
                acc_str = '{:.3f}'.format(acc)
            example_per_sec = self._log_freq * self._batch_size / max(duration, 1e-10)
            sec_per_batch = duration / self._log_freq
            format_str = ("{}: step {:d}, accuracy = {}, loss = {:.5f},"
                          " learning rate = {:.7f}({:.1f} examples/sec; {:.3f} sec/batch)")
            print(format_str.format(datetime.now(), self._step, acc_str, loss,
                                    lr, example_per_sec, sec_per_batch))


def step_lr(boundaries, values):
    step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant_decay(
        x=step,
        boundaries=boundaries,
        values=values)
    return learning_rate


def poly_lr(max_step, power=0.9):
    step = tf.train.get_or_create_global_step()
    learning_rate = tf.pow(1 - step / max_step, power)
    return learning_rate


def multi_lr(optimizer, loss, params, lr_multiplier):
    print(">>>>>>>>Set LR multiplier<<<<<<<<<<<<<")
    if not isinstance(lr_multiplier, dict):
        raise ValueError("`lr_multiplier` must has type dict,"
                         " but received: ", str(type(lr_multiplier)))
    if len(lr_multiplier) == 0:
        raise ValueError("`lr_multiplier` can not be empty")
    multiplied_grads_and_vars = []
    global_step = tf.train.get_or_create_global_step()

    def _get_multiplier(name):
        for key, value in lr_multiplier.items():
            if key in name:
                return lr_multiplier[key]
        return None

    grads_and_vars = optimizer.compute_gradients(loss, params)
    base_lr = getattr(optimizer, '_lr')
    none_counts = 0
    for grad, var in grads_and_vars:
        multiplier = _get_multiplier(var.op.name)
        if grad is None:
            none_counts += 1
        if multiplier is not None:
            if grad is None:
                raise ValueError('Requested multiple of `None` gradient.')
            if callable(multiplier):
                lr = multiplier(global_step, base_lr)
            elif not isinstance(multiplier, tf.Tensor):
                lr = tf.constant(multiplier) * base_lr
            else:
                lr = multiplier * base_lr
            if isinstance(grad, tf.IndexedSlices):
                tmp = grad.values * lr
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad *= lr
        multiplied_grads_and_vars.append((grad, var))
    if none_counts == len(multiplied_grads_and_vars):
        raise ValueError(
            "No gradients provided for any variable, check your graph for ops"
            " that do not support gradients, between variables %s and loss %s." %
            ([str(v) for _, v in grads_and_vars], loss))
    return optimizer.apply_gradients(multiplied_grads_and_vars, global_step=global_step)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


class RunMetaManager:

    @staticmethod
    def register_meta(root: str, run_cfg, env_cfg, *data_cfgs):
        mkdir(root)
        history_path = os.path.join(root, 'history.json')
        store_dir = os.path.join(root, run_cfg.model_name)
        mkdir(store_dir)
        model_dir = os.path.join(store_dir, 'model')
        mkdir(model_dir)
        test_dir = os.path.join(store_dir, 'test')
        mkdir(test_dir)
        run_cfg.update_hparam(dict(store_dir=store_dir,
                                   model_dir=model_dir,
                                   test_dir=test_dir))
        run_path = os.path.abspath(os.path.join(
            store_dir, run_cfg.name + '.json'))
        env_path = os.path.abspath(os.path.join(
            store_dir, env_cfg.name + '.json'))
        data_paths = [os.path.abspath(os.path.join(store_dir, data_config.name + '.json'))
                      for data_config in data_cfgs]
        run_cfg.to_json_file(run_path)
        env_cfg.to_json_file(env_path)
        for data_config, data_path in zip(data_cfgs, data_paths):
            data_config.to_json_file(data_path)
        # Timestamp
        with open(os.path.join(store_dir, 'timestamp.txt'), 'w') as f:
            f.write(str(datetime.now()))
        if os.path.exists(history_path):
            history = json.load(open(history_path))
            # Remove useless history
            keys = list(history.keys())
            for key in keys:
                if not os.path.exists(history[key][0]):
                    print('Clear useless {}'.format(key))
                    history.pop(key)
        else:
            history = dict()
        # Update history
        history[run_cfg.model_name] = [run_path, env_path] + data_paths
        json.dump(history,
                  open(history_path, 'w'), indent=2)
        print("Successfully updating history {} in to {})".format(
            run_cfg.model_name, history_path))

    @staticmethod
    def get_meta(model_name, root_dir: str):
        history_path = os.path.join(root_dir, 'history.json')
        if not os.path.exists(history_path):
            raise ValueError("Can not find history file, you may forget"
                             " invoking Executor.prepare to record config first.")
        history = json.load(open(history_path))
        if model_name not in history:
            raise ValueError("Can not find {}'s config in history records,"
                             "you may forget invoking Executor.prepare to "
                             "record {}'s config first.".format(model_name, model_name))
        print('==>Read configs from:\n' + '\n\t'.join(history[model_name]))
        return tuple(saving.from_json_file(path) for path in history[model_name])



class Executor:
    """
    Execution procedure:
    1. create hyper parameters templates and save into files
    2. read hyper parameters from files
    3. compile
    4. run
    """

    def __init__(self):
        self.executor = None
        self.steps = -1
        self.test_dir = None

    def compile(self, run_config, envir_config, model_fn):
        import logging
        import sys
        # Config logging
        tf.logging.set_verbosity(logging.INFO)
        handlers = [
            logging.FileHandler(os.path.join(run_config.store_dir, 'main.log')),
            logging.StreamHandler(sys.stdout)]
        logging.getLogger('tensorflow').handlers = handlers
        tf.logging.set_verbosity(tf.logging.INFO)
        self.test_dir = run_config.test_dir
        start = time.time()
        session_config = self._create_session_config(envir_config)
        exe_config = estimator.RunConfig(
            model_dir=run_config.model_dir,
            session_config=session_config,
            save_summary_steps=run_config.save_summary_steps,
            keep_checkpoint_max=run_config.keep_checkpoint_max,
            save_checkpoints_steps=run_config.save_checkpoints_steps,
            keep_checkpoint_every_n_hours=run_config.keep_checkpoint_every_n_hours)

        def _model_fn(features, labels, mode):
            if mode == estimator.ModeKeys.TRAIN:
                loss, accuracy, var_list, hooks = model_fn[mode](features, labels, run_config)
                # Learning rate
                # todo organize lr and optimizer configuration
                learning_rate = run_config.learning_rate
                if run_config.scheduler == 'exponential':
                    learning_rate = tf.train.exponential_decay(
                        learning_rate=learning_rate,
                        global_step=tf.train.get_or_create_global_step(),
                        decay_steps=run_config.decay_steps,
                        decay_rate=run_config.decay_rate,
                        staircase=run_config.staircase)
                elif run_config.scheduler == 'step':
                    learning_rate = step_lr(
                        boundaries=run_config.boundaries,
                        values=run_config.lr_values)
                else:
                    learning_rate = tf.constant(learning_rate, dtype=tf.float32)
                tf.summary.scalar('lr', learning_rate)
                # Optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                # Hook
                hooks += [LoggerHook(learning_rate=learning_rate,
                                     log_frequency=run_config.log_frequency,
                                     batch_size=run_config.batch_size,
                                     loss=loss, accuracy=accuracy,
                                     metric_names=run_config.class_names)]
                if hasattr(run_config, 'lr_multiplier'):
                    train_op = multi_lr(optimizer, loss, var_list, run_config.lr_multiplier)
                else:
                    train_op = optimizer.minimize(
                        loss, global_step=tf.train.get_global_step(), var_list=var_list)
                return estimator.EstimatorSpec(
                    estimator.ModeKeys.TRAIN, loss=loss,
                    training_hooks=hooks, train_op=train_op)
            elif mode == estimator.ModeKeys.EVAL:
                loss, metrics = model_fn[mode](features, labels, run_config)
                return estimator.EstimatorSpec(
                    estimator.ModeKeys.EVAL, loss=loss,
                    eval_metric_ops=metrics)
            elif mode == estimator.ModeKeys.PREDICT:
                predictions = model_fn[mode](features, run_config)
                return estimator.EstimatorSpec(
                    estimator.ModeKeys.PREDICT, predictions)
            else:
                raise ValueError("Expect mode in [train, eval, infer],"
                                 "but received {}".format(mode))

        self.executor = estimator.Estimator(
            model_fn=_model_fn,
            model_dir=run_config.model_dir,
            config=exe_config)
        self.steps = run_config.steps
        print(">>>>>>>>>>>>Finish Compiling in {:.2}s>>>>>>>>>>>>".format(time.time() - start))
        print(envir_config)
        print(run_config)
        flag = input('Is all config correct? (yes/no)')
        if flag not in ['yes', 'y', '1']:
            exit(-1)

    @staticmethod
    def _create_session_config(environment):
        # Config random
        random_seed = environment.random_seed
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        random.seed(random_seed)
        # Config GPU
        if hasattr(environment, 'CUDA_VISIBLE_DEVICES'):
            print(">>>>>>>>set device<<<<<<<<")
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = environment.CUDA_VISIBLE_DEVICES
        intra_op_parallelism_threads = environment.intra_op_parallelism_threads \
            if hasattr(environment, 'intra_parallelism_threads') else 0
        inter_op_parallelism_threads = environment.inter_op_parallelism_threads \
            if hasattr(environment, 'inter_parallelism_threads') else 0
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=inter_op_parallelism_threads,
            intra_op_parallelism_threads=intra_op_parallelism_threads)
        if hasattr(environment, 'per_process_gpu_memory_fraction'):
            session_config.gpu_options.per_process_gpu_memory_fraction = \
                environment.per_process_gpu_memory_fraction
        if hasattr(environment, 'allow_growth'):
            session_config.gpu_options.allow_growth = environment.allow_growth
        return session_config

    def run(self,
            input_fn,
            eval_input_fn=None,
            mode=estimator.ModeKeys.TRAIN,
            visual_func=None,
            early_stop_steps=500,
            early_stop_begin_step=0,
            early_stop_interval=60,
            early_stop_metric='accuracy'):
        if input_fn is None:
            raise ValueError("Input function must be specified")
        if mode == estimator.ModeKeys.PREDICT:
            if visual_func is None:
                raise ValueError("When inference, you must specify a"
                                 " visualization function.")
            print('=> Inference')
            tmp = list(self.executor.predict(input_fn=input_fn))
            print(tmp)
            visual_func(tmp, self.test_dir)
        elif mode == estimator.ModeKeys.EVAL:
            print("=> Evaluating")
            self.executor.evaluate(input_fn=input_fn)
        elif mode == estimator.ModeKeys.TRAIN:
            if eval_input_fn is not None:
                assert early_stop_metric in ['f1', 'accuracy', 'loss']
                print("=> Train while evaluating")
                hook = estimator.stop_if_no_increase_hook(
                    self.executor, early_stop_metric, max_steps_without_increase=early_stop_steps,
                    min_steps=early_stop_begin_step, run_every_secs=early_stop_interval)
                train_spec = estimator.TrainSpec(input_fn=input_fn, hooks=[hook])
                eval_spec = estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=early_stop_interval)
                estimator.train_and_evaluate(self.executor, train_spec, eval_spec)
            else:
                print("=> Training")
                self.executor.train(input_fn=input_fn, steps=self.steps, max_steps=None)
        else:
            raise ValueError("Expected value of 'mode' in [infer, eval, train], but received ", mode)
