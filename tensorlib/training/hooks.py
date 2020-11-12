from tensorlib.training.sessions.core import SessRunHook, SessRunArgs, SaverTrigger
from tensorlib.utils import ProgressBar
from tensorflow.python.framework import meta_graph
import math
import tensorflow as tf
import numpy as np
import warnings
import os
import time
from datetime import datetime


class StepHook(object):
    """
    self._every_steps is like a time period and self._last_triggered_step is like point in time
       t_s                    t_s                      t_s
        |______________________|________________________|
              every_steps             every_steps

    when should_trigger_for_step() return True, Training will stop and call evaluation function to
    evaluate
    """

    def __init__(self, every_steps=None, every_times=None):
        self._last_triggered_step = None
        self._last_triggered_time = None
        self._every_steps = every_steps
        self._every_times = every_times

    def reset(self):
        self._last_triggered_step = None
        self._last_triggered_time = None

    def should_trigger_for_step(self, step):
        if self._last_triggered_step is None:
            return True

        if self._last_triggered_step == step:
            return False

        if self._every_times is not None:
            if time.time() > self._last_triggered_time + self._every_times:
                return True

        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True
        return False

    def update_last_triggered_step(self, step):
        current_time = time.time()
        if self._last_triggered_time is None:
            elapsed_steps = None
            elapsed_times = None
        else:
            elapsed_times = current_time - self._last_triggered_time
            elapsed_steps = step - self._last_triggered_step
        self._last_triggered_time = current_time
        self._last_triggered_step = step
        return elapsed_times, elapsed_steps

    def last_triggered_step(self):
        return self._last_triggered_step

    def set_every_steps(self, every_steps):
        self._every_steps = every_steps


class SummaryHook(SessRunHook):

    def __init__(self,
                 save_steps=None,
                 save_secs=None,
                 summary_op=None,
                 output_dir=None,
                 summary_writer=None,
                 reset_step=False):
        self._summary_op = summary_op
        self._summary_writer = summary_writer
        self._request_summary = None
        self._next_step = None
        self._global_step_tensor = None
        self._output_dir = output_dir
        self._reset_step = reset_step
        self._save_steps = save_steps
        self._timer = StepHook(
            every_times=save_secs, every_steps=save_steps)

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use SummaryHook.")

    def before_run(self, run_context):
        if not self._reset_step:
            self._request_summary = (
                self._next_step is None or
                self._timer.should_trigger_for_step(self._next_step))
        else:
            self._request_summary = (
                    self._next_step is None or
                    (self._next_step == self._save_steps))

        requests = {}
        if self._request_summary:
            if self._get_summary_op() is not None:
                requests["summary"] = self._get_summary_op()
        return SessRunArgs(requests)

    def after_run(self, run_context, run_values):
        _ = run_context
        if not self._summary_writer:
            return
        global_step = run_context.session.run(self._global_step_tensor)

        if not self._reset_step:
            if self._request_summary:
                self._timer.update_last_triggered_step(global_step)
                if "summary" in run_values.results:
                    for summary in run_values.results["summary"]:
                        self._summary_writer.add_summary(summary, global_step)
            self._next_step = global_step + 1
        else:
            if self._request_summary:
                if "summary" in run_values.results:
                    for summary in run_values.results["summary"]:
                        self._summary_writer.add_summary(summary, global_step)
                self._next_step = 1
            else:
                self._next_step += 1

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()

    def _get_summary_op(self):
        summary_op = None
        if self._summary_op is not None:
            summary_op = self._summary_op

        if summary_op is None:
            return None

        if not isinstance(summary_op, list):
            return [summary_op]
        return summary_op


class EarlyStoppingHook(SessRunHook):

    def __init__(self, loss_name, feed_dict=None, tolerance=0.001, stopping_step=10000):
        """
        description: 'EarlyStoppingHook' can be used in the same way as the predefined
        'StopAtStepHook' it can be passed to the hooks list of the monitored training
        sessions
        Arg:
            loss_name:
            feed_dict:
            tolerance:
            stopping_step:
        """
        self.loss_name = loss_name
        if feed_dict is None:
            self.feed_dict = {}

        self.tolerance = tolerance
        self.stopping_step = stopping_step
        self._global_step = None
        self._step = 0

    def begin(self):
        self._global_step = tf.train.get_global_step()
        if self._global_step is None:
            raise RuntimeError("Global step must be defined!")
        self._step = 0

    def before_run(self, run_context):
        """
        Specify feed_dict and tensors to be evaluated
        the sessions and graph are accessible through the 'run_context'
        argument passed to hooks methods from monitored sessions
        Additional tensors in the graph can be retrieved using:
            graph = run_context.sessions.graph
            tensor = graph.get_tensor_by_name("name")
        """
        if self._step % self.stopping_step == 0:
            graph = run_context.session.graph
            loss = graph.get_tensor_by_name(self.loss_name)
            fd = {}
            for key, value in self.feed_dict.items():
                placeholder = graph.get_tensor_by_name(key)
                fd[placeholder] = value
            return SessRunArgs({"step": self._global_step,
                                "loss": loss}, feed_dict=fd)
        else:
            return SessRunArgs({"step": self._global_step})

    def after_run(self, run_context, run_values):
        """
            Check if current loss is below tolerance
        Arg：
            The monitored training sessions passes the values of fetches to the
            'run_values' argument of 'after_run()' in a dictionary .
            Stop request are sent using: 'run_context.request_stop()'
        """
        if self._step % self.stopping_step == 0:
            global_step = run_values.results["step"]
            current_loss = run_values.results["loss"]
            if current_loss < self.tolerance:
                run_context.request_stop()
        else:
            global_step = run_values.results["step"]
        self._step = global_step


class NanTensorHook(SessRunHook):
    def __init__(self, loss_tensor, fail_on_nan_loss=True):
        self._loss_tensor = loss_tensor
        self._fail_on_nan_loss = fail_on_nan_loss

    def before_run(self, run_context):
        return SessRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        if np.isnan(run_values):
            failure_message = "Model diverged with loss = NaN."
            if self._fail_on_nan_loss:
                raise RuntimeError("NaN loss during training.")
            else:
                warnings.warn(failure_message)
                run_context.request_stop()


class Hdf5SaverHook(SessRunHook):
    pass


class CkPtSaverHook(SessRunHook):
    def __init__(self,
                 file_dir,
                 file_basename="model.ckpt",
                 save_steps=None,
                 save_secs=None,
                 saver=None,
                 compiler=None,
                 global_step_tensor=None,
                 saver_triggers=None):
        self._saver = saver
        self._file_dir = file_dir
        self._save_path = os.path.join(file_dir, file_basename)
        self._compiler = compiler
        self._timer = StepHook(
            every_steps=save_steps, every_times=save_secs)
        self._summary_writer = None
        self._global_step_tensor = global_step_tensor
        self._saver_triggers = saver_triggers or []
        self._steps_per_run = 1

    def set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def add_saver_triggers(self, saver_triggers):
        if not isinstance(saver_triggers, list):
            saver_triggers = [saver_triggers]
        self._saver_triggers.extend(saver_triggers)

    def begin(self):
        if self._summary_writer is None:
            self._summary_writer = tf.summary.FileWriterCache.get(self._file_dir)
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use CkPtSaverHook.")
        for s in self._saver_triggers:
            s.begin()

    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        graph = tf.get_default_graph()
        saver_def = self._get_saver().saver_def if self._get_saver() else None
        meta_graph_def = meta_graph.create_meta_graph_def(
            graph_def=graph.as_graph_def(add_shapes=True),
            saver_def=saver_def)
        self._summary_writer.add_graph(graph)
        self._summary_writer.add_meta_graph(meta_graph_def)
        self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):
        return SessRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        previous_global_step = run_values.results
        if self._timer.should_trigger_for_step(previous_global_step + self._steps_per_run):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                if self._save(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self._save(session, last_step)
        for s in self._saver_triggers:
            s.end(session, last_step)

    def _get_saver(self):
        if self._saver is not None:
            return self._saver
        elif self._compiler is not None:
            return self._compiler.saver

        # Get saver from the SAVERS collection if present.
        collection_key = tf.GraphKeys.SAVERS
        savers = tf.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                "No items in collection {}. Please add a saver to the collection "
                "or provide a saver or compiler.".format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor."
                .format(collection_key))
        self._saver = savers[0]
        return savers[0]

    def _save(self, session, step):
        for s in self._saver_triggers:
            s.before_save(session, step)
        self._get_saver().save(session, self._save_path, global_step=step)
        should_stop = False
        for s in self._saver_triggers:
            if s.after_save(session, step):
                should_stop = True
        return should_stop


class EvalHook(SessRunHook):
    def __init__(self, num_steps=None, is_logging=True):
        self._num_steps = num_steps
        self._is_logging = is_logging
        self._logging_frq = 1 if num_steps is None or num_steps < 20 else math.floor(num_steps / 10)
        self._evaluated_num = 0

    def update_evaluated_num(self, update_eval_step):
        """ update_eval_step: A tensor"""
        self._evaluated_num = update_eval_step

    def before_run(self, run_context):
        return SessRunArgs({"evaluated_num": self._evaluated_num})

    def after_run(self,
                  run_context,
                  run_values):
        evaluated_num = run_values.results["evaluated_num"]
        if self._is_logging:
            if self._num_steps is None:
                tf.logging.info("")
            else:
                if evaluated_num % self._logging_frq == 0 or self._num_steps == evaluated_num:
                    tf.logging.info("")
        if self._num_steps is not None and evaluated_num >= self._num_steps:
            run_context.request_stop()


class EvaluateSaverTrigger(SaverTrigger):
    def __init__(self, evaluator, every_triggers_step):
        self._evaluator = evaluator
        self._every_triggers_step = every_triggers_step
        self._step_hook = StepHook()

        self._eval_result = None
        self._is_first_run = False

    def _evaluate(self, global_step):
        self._step_hook.update_last_triggered_step(global_step)
        self._eval_result = self._evaluator()

    @property
    def eval_result(self):
        return self._eval_result

    # executed begin of using session
    def begin(self):
        # Evaluate every triggers step
        self._step_hook.set_every_steps(self._every_triggers_step)
        self._is_first_run = True

    # executed after call Saver.save()
    # can request training to be stopped, by returning True in `after_save`
    def after_save(self, session, global_step):
        del session
        if self._is_first_run:
            self._is_first_run = False
            return False

        if self._step_hook.should_trigger_for_step(global_step):
            self._evaluate(global_step)
            # TODO add early stop mechanism to stop train i.e. return True
        else:
            return False

    # executed at the end of session
    def end(self, session, global_step):
        if global_step != self._step_hook.last_triggered_step():
            self._evaluate(global_step)


class ProgressHook(SessRunHook):

    def __init__(self,
                 title,
                 epochs,
                 metric_names,
                 target,
                 initial_epoch=0,
                 stateful_metric_names=None):
        super(ProgressHook, self).__init__()
        if target is None:
            raise ValueError("`target`(total steps) must be set")
        if stateful_metric_names:
            self._stateful_metric_names = set(stateful_metric_names)
        else:
            self._stateful_metric_names = set()
        self._title = title
        self._epochs = epochs
        self._initial_epoch = initial_epoch
        self._epoch = 1
        self._progbar = None
        self._target = target
        self._seen = None
        self._log_values = None
        self._metric_names = metric_names

    def begin(self):
        self._seen = 0

    def before_run(self, run_context):
        if self._seen == 0:  # on epoch begin else on batch begin
            print(self._title + ' Epoch %d/%d' % (self._epoch + self._initial_epoch, self._epochs))
            self._progbar = ProgressBar(target=self._target,
                                        stateful_metrics=self._stateful_metric_names)
        if self._seen < self._target:
            self._log_values = []
        metrics = run_context.original_args.fetches[0][:len(self._metric_names)]
        # print(metrics)
        return SessRunArgs(metrics)

    def after_run(self, run_context, run_values):
        metrics = run_values.results
        for name, metric in zip(self._metric_names, metrics):
            self._log_values.append((name, metric))
        self._seen += 1
        self._progbar.update(self._seen, self._log_values)
        if self._seen >= self._target:  # on epoch end else on batch end
            self._seen = 0
            self._epoch += 1


class PredictHook(SessRunHook):
    def __init__(self,
                 outputs,
                 predict_fn):
        self._outputs = outputs
        self._step = 0
        self._predict_fn = predict_fn

    def begin(self):
        self._step = 1

    def before_run(self, run_context):
        return SessRunArgs(self._outputs)

    def after_run(self, run_context, run_values):
        outputs = run_values.results
        self._predict_fn(outputs, self._step)


class LoggerHook(SessRunHook):
    def __init__(self,
                 title,
                 batch_size,
                 log_frequency=1,
                 metric_names=None):
        self.title = title
        self._batch_size = batch_size
        self._log_frequency = log_frequency
        self._metric_names = metric_names
        self._step = 0
        self._start_time = None

    def begin(self):
        self._step = 1
        self._start_time = time.time()

    def before_run(self, run_context):
        metrics = run_context.original_args.fetches[:len(self._metric_names)]
        print(self._metric_names, run_context.original_args.fetches)
        self._step += 1
        return SessRunArgs(metrics)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        if self._step % self._log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            metrics = run_values.results
            example_per_sec = self._log_frequency * self._batch_size / max(duration, 1e-10)
            sec_per_batch = duration / self._log_frequency
            metric_str = ', '.join(name + ' = ' + str(metric)
                                   for name, metric in zip(self._metric_names, metrics))
            format_str = self.title + " {}: step {:d} {} ({:.1f} examples/sec; {:.3f} sec/batch)"
            print(format_str.format(datetime.now(), self._step, metric_str, example_per_sec, sec_per_batch))
