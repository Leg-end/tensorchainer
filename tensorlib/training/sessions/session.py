from tensorlib.training.sessions.core import SessRunArgs, SessRunContext, SessRunValues
from tensorlib.training.sessions.core import BasicSessionCreator
from tensorlib.engine.base_lib import get_session, is_tensor
from tensorflow.python.client.session import _FetchHandler
from tensorflow.python.util import function_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops as tf_ops
import tensorflow as tf
import numpy as np
import six
import sys


class _WrappedSession(object):
    def __init__(self, sess):
        self._sess = sess
        self._should_stop = False
        self._wrapped_is_stoppable = isinstance(self._sess, _WrappedSession)

    @property
    def graph(self):
        return self._sess.graph

    def should_stop(self):
        if self._should_stop:
            return True
        if self._sess:
            return self._wrapped_is_stoppable and self._sess.should_stop()
        # if self._sess is None(i.e session close) will stop
        return True

    def close(self):
        if self._sess:
            try:
                self._sess.close()
            except(tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                print("error %s!!!" % e)

            finally:
                self._sess = None

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)

    def run_step_fn(self, step_fn, raw_session, run_with_hooks):
        run_with_hooks = run_with_hooks or self.run
        return step_fn(StepContext(raw_session, run_with_hooks))


class StepContext(object):
    def __init__(self, session, run_with_hooks_fn):
        self._session = session
        self._run_with_hooks_fn = run_with_hooks_fn

    @property
    def session(self):
        return self._session

    def run_with_hooks(self, *args, **kwargs):
        return self._run_with_hooks_fn(*args, **kwargs)

    def request_stop(self):
        raise StopIteration('step_fn has requested the iterations to stop.')


class _HookedSession(_WrappedSession):
    def __init__(self, sess, hooks):
        _WrappedSession.__init__(self, sess)
        self._hooks = hooks
        self._should_stop = False

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        if self.should_stop():
            raise RuntimeError("Run called even after should_stop requested.")
        actual_fetches = {'caller': fetches}
        run_context = SessRunContext(
            original_args=SessRunArgs(fetches, feed_dict),
            session=self._sess)

        feed_dict = self._call_hook_before_run(run_context, actual_fetches,
                                               feed_dict)

        outputs = _WrappedSession.run(
            self, fetches=actual_fetches, feed_dict=feed_dict, run_metadata=run_metadata, options=options)

        for hook in self._hooks:
            hook.after_run(
                run_context,
                SessRunValues(
                    results=outputs[hook] if hook in outputs else None,
                    run_metadata=run_metadata))
        self._should_stop = self._should_stop or run_context.stop_requested
        return outputs["caller"]

    def _call_hook_before_run(self, run_context, actual_fetches, user_feed_dict):
        hook_feeds = {}
        for hook in self._hooks:
            # return SessRunArg
            request = hook.before_run(run_context)
            if request is not None:
                if not isinstance(request, SessRunArgs):
                    raise TypeError("")
                if request.fetches is not None:
                    actual_fetches[hook] = request.fetches
                if request.feed_dict:
                    self._raise_if_feeds_intersects(hook_feeds, request.feed_dict,
                                                    'Same tensor is fed by two hooks.')
                    hook_feeds.update(request.feed_dict)
        if not hook_feeds:
            return user_feed_dict
        if not user_feed_dict:
            return hook_feeds

        self._raise_if_feeds_intersects(
            user_feed_dict, hook_feeds,
            'Same tensor is fed by a SessionRunHook and user.')
        hook_feeds.update(user_feed_dict)
        return hook_feeds

    @staticmethod
    def _raise_if_feeds_intersects(hook_feeds, feed_dicts, message):
        intersection = set(hook_feeds.keys()) & set(feed_dicts.keys())
        if intersection:
            raise RuntimeError(message + ' Conflict(s): ' + str(list(intersection)))


class _CoordSession(_WrappedSession):
    def __init__(self, sess, coord, stop_grace_period_secs=120):
        _WrappedSession.__init__(self, sess)
        self._coord = coord
        self._stop_grace_period_secs = stop_grace_period_secs

    def close(self):
        self._coord.request_stop()
        try:
            self._coord.join(
                stop_grace_period_secs=self._stop_grace_period_secs,
                ignore_live_threads=True)
        finally:
            try:
                _WrappedSession.close(self)
            except Exception:
                pass

    def run(self, *args, **kwargs):
        try:
            return self._sess.run(*args, **kwargs)
        except (tf.errors.AbortedError, tf.errors.UnavailableError):
            raise
        except Exception:
            original_exc_info = sys.exc_info()
            try:
                self._coord.raise_requested_exception()
            except (tf.errors.AbortedError, tf.errors.UnavailableError):
                raise
            except Exception:
                raise six.reraise(*original_exc_info)
            else:
                raise six.reraise(*original_exc_info)


class Session(object):
    def __init__(self,
                 session_creator=None,
                 hooks=None,
                 stop_grace_period_secs=120):
        self._graph_was_finalized = tf.get_default_graph().finalized
        self._hooks = hooks or []

        for hook in self._hooks:
            hook.begin()

        self._session_creator = session_creator or BasicSessionCreator()
        self._stop_grace_period_secs = stop_grace_period_secs

        self.raw_sess = self._session_creator.create_session()

        self.coord = tf.train.Coordinator()
        if tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            tf.train.start_queue_runners(self.raw_sess, self.coord)

        for hook in self._hooks:
            hook.after_create_session(self.raw_sess, self.coord)

        self._sess = _CoordSession(
            sess=_HookedSession(sess=self.raw_sess, hooks=self._hooks), coord=self.coord,
            stop_grace_period_secs=self._stop_grace_period_secs)

    def graph(self):
        return self._sess.graph

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self._sess.run(
            fetches=fetches,
            feed_dict=feed_dict,
            options=options,
            run_metadata=run_metadata)

    def run_step_fn(self, step_fn):
        step_fn_arguments = function_utils.fn_args(step_fn)
        if step_fn_arguments != ('step_context',) and step_fn_arguments != (
                'self', 'step_context'):
            raise ValueError(
                '`step_fn` may either have one `step_context` argument, or'
                ' `self` and `step_context` arguments if it\'s an instance'
                ' method. Got {} instead.'.format(step_fn_arguments))
        return self._sess.run_step_fn(step_fn, self.raw_sess, run_with_hooks=None)

    def should_stop(self):
        return self._sess is None or self._sess.should_stop()

    def close(self):
        self._close_internal()

    def _close_internal(self, exception_type=None):
        try:
            if not exception_type:
                for h in self._hooks:
                    h.end(self.raw_sess)
        finally:
            try:
                if self._sess is None:
                    raise RuntimeError("Session is already closed.")
                self._sess.close()
            finally:
                self._sess = None
                self.coord = None
                if not self._graph_was_finalized:
                    tf.get_default_graph()._unsafe_unfinalize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type in [tf.errors.OutOfRangeError, StopIteration]:
            exc_type = None
        self._close_internal(exc_type)
        return exc_type is None


class Function(object):
    @property
    def should_stop(self):
        return self._should_stop

    def __init__(self, inputs, outputs,
                 updates=None,
                 **kwargs):
        if not updates:
            updates = []
        assert isinstance(inputs, (list, tuple))
        assert isinstance(outputs, (list, tuple))
        assert isinstance(updates, (list, tuple))
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else:
                    updates_ops.append(update)
            self.updates_ops = tf.group(*updates_ops)
        self.feed_dict = kwargs.pop("feed_dict", {})
        # The main use case of `fetches` being passed to a model is the ability
        # to run custom updates
        # (since the outputs of fetches are never returned).
        # This requires us to wrap fetches in `identity` ops.
        self.fetches = [tf.identity(x) for x in kwargs.pop("fetches", [])]
        self.hooks = kwargs.pop("hooks", [])
        if not isinstance(self.hooks, list):
            self.hooks = [self.hooks]

        for h in self.hooks:
            h.begin()

        self.tf_sess = get_session()
        for hook in self.hooks:
            hook.after_create_session(self.tf_sess, None)
        self.run_options = kwargs.pop('options', None)
        self.run_metadata = kwargs.pop('run_metadata', None)
        self.graph = kwargs.pop('graph', None)
        self._should_stop = False
        self._callable_fn = None
        self._feed_arrays = None
        self._feed_symbols = None
        self._symbol_values = None
        self._fetch_handler = None
        self._all_fetches = {}
        self.hook_feed_dicts = {}

    def _make_callable(self, feed_arrays, feed_symbols, symbol_values, all_fetches):
        callable_opts = config_pb2.CallableOptions()
        for x in feed_arrays:
            callable_opts.feed.append(x.name)
        if self.feed_dict:
            for key in sorted(self.feed_dict.keys()):
                callable_opts.feed.appned(key.name)

        for x, y in zip(feed_symbols, symbol_values):
            connection = callable_opts.tensor_connection.add()
            if x.dtype != y.dtype:
                y = tf.cast(y, x.dtype)
            from_tensor = tf_ops._as_graph_element(y)
            if from_tensor is None:
                from_tensor = y
            connection.from_tensor = from_tensor.name
            connection.to_tensor = x.name

        self._all_fetches = all_fetches

        self._fetch_handler = _FetchHandler(
            graph=self.graph or tf.get_default_graph(),
            fetches=self._all_fetches, feeds={})
        for x in self._fetch_handler.fetches():
            callable_opts.fetch.append(x.name)

        callable_opts.target.append(self.updates_ops.name)

        if self.run_options:
            callable_opts.run_options.CopyFrom(self.run_options)
        callable_fn = self.tf_sess._make_callable_from_options(callable_opts)

        self._callable_fn = callable_fn
        self._feed_arrays = feed_arrays
        self._feed_symbols = feed_symbols
        self._symbol_values = symbol_values

    def call(self, inputs, all_fetches):
        assert isinstance(inputs, (list, tuple))
        feed_symbols = []
        symbol_values = []
        feed_arrays = []
        array_values = []
        for tensor, value in zip(self.inputs, inputs):
            if value is None:
                continue
            if is_tensor(value):
                feed_symbols.append(tensor)
                symbol_values.append(value)
            else:
                feed_arrays.append(tensor)
                array_values.append(
                    np.asarray(value, dtype=tf.as_dtype(tensor.dtype).as_numpy_dtype))
        if self.feed_dict:
            for key in sorted(self.feed_dict.keys()):
                array_values.append(
                    np.asarray(self.feed_dict[key], dtype=tf.as_dtype(key.dtype).as_numpy_dtype))
        if (self._callable_fn is None or
            feed_arrays != self._feed_arrays or
            symbol_values != self._symbol_values or
            feed_symbols != self._feed_symbols or
                all_fetches != self._all_fetches):
            self._make_callable(feed_arrays=feed_arrays,
                                feed_symbols=feed_symbols,
                                symbol_values=symbol_values,
                                all_fetches=all_fetches)
        if self.run_metadata:
            fetched = self._callable_fn(*array_values, run_metadata=self.run_metadata)
        else:
            fetched = self._callable_fn(*array_values)
        return fetched

    def run(self, inputs):
        all_fetches = {"output": self.outputs,
                       "fetches": self.fetches}
        run_context = SessRunContext(
            original_args=SessRunArgs((self.outputs, self.fetches), self.feed_dict),
            session=self.tf_sess)

        all_fetches.update(self._call_hook_before_run(run_context))

        outputs = self.call(inputs, all_fetches)
        outputs = self._fetch_handler.build_results(self.tf_sess, outputs)
        for hook in self.hooks:
            hook.after_run(
                run_context,
                SessRunValues(
                    results=outputs[hook] if hook in outputs else None,
                    run_metadata=self.run_metadata))
        self._should_stop = self._should_stop or run_context.stop_requested
        return outputs["output"]

    def __call__(self, inputs):
        try:
            return self.run(inputs)
        except tf.errors.OutOfRangeError:
            self._should_stop = True
            print("Stop due to end of data.")

    def _call_hook_before_run(self, run_context):
        hook_feeds = {}
        actual_fetches = {}
        for hook in self.hooks:
            request = hook.before_run(run_context)
            if request is not None:
                if not isinstance(request, SessRunArgs):
                    raise TypeError("Expect `request` is an instance of SessRunArgs"
                                    " but received: %s from hook %s" % (str(type(request)), str(hook)))
                if request.fetches is not None:
                    actual_fetches[hook] = request.fetches
                if request.feed_dict:
                    self._raise_if_feeds_intersects(hook_feeds, request.feed_dict,
                                                    'Same tensor is fed by two hooks.')
                    hook_feeds.update(request.feed_dict)

        self._raise_if_feeds_intersects(
            self.hook_feed_dicts, hook_feeds,
            'Same tensor is fed by a SessionRunHook and user.')
        self.hook_feed_dicts.update(hook_feeds)
        return actual_fetches

    @staticmethod
    def _raise_if_feeds_intersects(hook_feeds, feed_dicts, message):
        intersection = set(hook_feeds.keys()) & set(feed_dicts.keys())
        if intersection:
            raise RuntimeError(message + ' Conflict(s): ' + str(list(intersection)))

    def end(self):
        try:
            for h in self.hooks:
                h.end(self.tf_sess)
        except Exception as e:
            print(e.args)
        finally:
            pass


def _train_pipeline():
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(np.arange(100.).reshape(
            [100, 1]), dtype=tf.float32))
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    element = iterator.get_next()
    return element


def _eval_pipeline():
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(np.arange(100., 200).reshape(
            [100, 1]), dtype=tf.float32))
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    element = iterator.get_next()
    return element


# if __name__ == '__main__':
#     from tensorlib.training.hooks import CkPtSaverHook
#     train_data = _train_pipeline()
#     global_steps = tf.Variable(0)
#     update_train_steps = tf.assign_add(global_steps, 1)
#     hooks = CkPtSaverHook(
#         file_dir='./test_ckpt',
#         global_step_tensor=global_steps,
#         save_steps=5)
#     val_data = _eval_pipeline()
#     inputs = tf.identity(train_data, "inputs")
#     outputs = tf.add(inputs, 1)
#     fetches = tf.constant(222)
#     f = Function(inputs=[inputs], outputs=[outputs], updates=[update_train_steps], hooks=[hooks],
#                  fetches={"custom": fetches})
#     val_f = Function(inputs=[inputs], outputs=[outputs])
#     for i in range(1, 25):
#         print(f([train_data]))
#         if i % 5 == 0:
#             print(">>>>>>>start eval>>>>>>>>>")
#             for j in range(5):
#                 print(val_f([val_data]))
#             print(">>>>>>>end eval>>>>>>>>>")
