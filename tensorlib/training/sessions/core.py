from tensorflow.python.ops import resources
import tensorflow as tf
import abc
import collections


class SessRunArgs(collections.namedtuple("RunArgs", ['fetches', 'feed_dict'])):
    def __new__(cls, fetches, feed_dict=None):
        return super(SessRunArgs, cls).__new__(cls, fetches, feed_dict)


class SessRunContext(object):
    def __init__(self, original_args, session):
        self._original_args = original_args
        self._session = session
        self._stop_requested = False

    @property
    def original_args(self):
        return self._original_args

    @property
    def session(self):
        return self._session

    @property
    def stop_requested(self):
        return self._stop_requested

    def request_stop(self):
        self._stop_requested = True


class SessRunValues(collections.namedtuple("SessRunValues", 'results, run_metadata')):
    pass


class SessRunHook(object):
    """
    Description of 'SessRunHook':
        The pseudocode detailing the execution order is as follows:
            'call' hooks.begin()
            sess = tf.Session()
            'call' hooks.after_create_session()
            while not stop is requested:
                'call' hooks.before_run()
                try:
                    results = sess.run(merged_fetches, feed_dict=merged_feeds)
                except (errors.OutOfRangeError, StopIteration):
                    break
                'call' hooks.after_run()
            'call' hooks.end()
            sess.close()
    """

    def begin(self):
        pass

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        return None

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass


class _SessionManager(object):
    def __init__(self,
                 local_init_op=None,
                 graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        self._graph = graph
        self._local_init_op = local_init_op

    def _restore_checkpoint(self,
                            saver=None,
                            checkpoint_dir=None,
                            checkpoint_path=None,
                            sess_config=None):
        sess = tf.Session(graph=self._graph, config=sess_config)
        if checkpoint_dir and checkpoint_path:
            raise ValueError("Can not provide both checkpoint_dir and "
                             "checkpoint_path.")

        if not saver or not (checkpoint_dir or checkpoint_path):
            return sess, False

        if checkpoint_path:
            saver.restore(sess, checkpoint_path)
            return sess, True

        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        while not checkpoint or not checkpoint.model_checkpoint_path:
            return sess, False

        saver.restore(sess, checkpoint.model_checkpoint_path)
        saver.recover_last_checkpoints(checkpoint.all_model_checkpoint_paths)
        return sess, True

    def get_session(self,
                    init_op=None,
                    init_fn=None,
                    checkpoint_dir=None,
                    checkpoint_path=None,
                    init_feed_dict=None,
                    config=None,
                    saver=None):
        sess, is_loaded_from_ckpt = self._restore_checkpoint(
            saver=saver, checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path, sess_config=config)
        if not is_loaded_from_ckpt:
            if init_op is None and not init_fn and self._local_init_op is None:
                raise RuntimeError("Model is not initialized and no init_op or "
                                   "init_fn or local_init_op was given")
            if init_op is not None:
                sess.run(init_op, feed_dict=init_feed_dict)
            if init_fn:
                init_fn(sess)
        return sess


class SaverTrigger(object):
    def begin(self):
        pass

    def before_save(self, session, global_step_value):
        pass

    def after_save(self, session, global_step_value):
        pass

    def end(self, session, global_step_value):
        pass


class EvaluateSaverTrigger(SaverTrigger):
    def __init__(self, evaluator, every_triggers_step=None):

        from tensorlib.training.hooks import StepHook
        self._evaluator = evaluator
        self._every_triggers_step = every_triggers_step
        self._step_hook = StepHook()

        self._eval_result = None
        self._is_first_run = False

    def _evaluate(self, global_step):
        self._step_hook.update_last_triggered_step(global_step)
        self._eval_result = self._evaluator()
        if self._eval_result.status != "evaluated":
            raise RuntimeError("There wa no new checkpoint after the training. eval status: {}"
                               .format(self._eval_result.status))

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
            return

        if self._step_hook.should_trigger_for_step(global_step):
            self._evaluate(global_step)
            return True    # will request training to be stopped
        else:
            return False

    # executed at the end of session
    def end(self, session, global_step):
        if global_step != self._step_hook.last_triggered_step():
            self._evaluate(global_step)


class SessionCreator(object):

    @abc.abstractmethod
    def create_session(self):
        raise NotImplementedError(
            'create_session is not implemented for {}.'.format(self))


class BasicSessionCreator(SessionCreator):
    def __init__(self,
                 compiler=None,
                 config=None,
                 checkpoint_dir=None,
                 checkpoint_path=None):
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_path = checkpoint_path
        self._compiler = compiler or Compiler()
        self._config = config
        self._session_manager = None

    def _get_session_manager(self):
        if self._session_manager:
            return self._session_manager

        self._session_manager = _SessionManager(
            local_init_op=self._compiler.local_init_op,
            graph=tf.get_default_graph())
        return self._session_manager

    def create_session(self):
        self._compiler.finalize()
        return self._get_session_manager().get_session(
            saver=self._compiler.saver,
            checkpoint_dir=self._checkpoint_dir,
            checkpoint_path=self._checkpoint_path,
            config=self._config,
            init_op=self._compiler.init_op,
            init_feed_dict=self._compiler.init_feed_dict,
            init_fn=self._compiler.init_fn)


class Compiler(object):
    def __init__(self,
                 init_op=None,
                 saver=None,
                 init_fn=None,
                 init_feed_dict=None,
                 local_init_op=None,
                 ready_op=None,
                 ready_for_local_init_op=None):
        self._init_op = init_op
        if init_fn:
            self._init_fn = lambda sess: init_fn(self, sess)
        else:
            self._init_fn = None

        self._saver = saver
        self._local_init_op = local_init_op
        self._init_feed_dict = init_feed_dict
        self._ready_op = ready_op
        self._ready_for_local_init_op = ready_for_local_init_op

    def finalize(self):
        if self._init_op is None:
            self._init_op = Compiler.get_or_default(
                "init_op", tf.GraphKeys.INIT_OP, default_init_op)

        if self._ready_op is None:
            self._ready_op = Compiler.get_or_default(
                "ready_op", tf.GraphKeys.READY_OP, default_ready_op)

        if self._ready_for_local_init_op is None:
            self._ready_for_local_init_op = Compiler.get_or_default(
                "_ready_for_local_init_op", tf.GraphKeys.READY_FOR_LOCAL_INIT_OP,
                default_ready_for_local_init_op)

        if self._local_init_op is None:
            self._local_init_op = Compiler.get_or_default(
                "local_init_op", tf.GraphKeys.LOCAL_INIT_OP, default_local_init_op)

        if self._saver is None:
            self._saver = get_saver_or_default()
        return self

    @staticmethod
    def get_or_default(arg_name, collection_key, default_constructor):
        elements = tf.get_collection(collection_key)
        if elements:
            if len(elements) > 1:
                raise RuntimeError(
                    'More than one item in the collection "%s". '
                    'Please indicate which one to use by passing it to '
                    'the Compiler as:  Compiler(%s=item to use)', collection_key, arg_name)
            return elements[0]
        op = default_constructor()
        if op is not None:
            tf.add_to_collection(collection_key, op)
        return op

    @property
    def init_fn(self):
        return self._init_fn

    @property
    def init_op(self):
        return self._init_op

    @property
    def ready_op(self):
        return self._ready_op

    @property
    def ready_for_local_init_op(self):
        return self._ready_for_local_init_op

    @property
    def local_init_op(self):
        return self._local_init_op

    @property
    def saver(self):
        return self._saver

    @property
    def init_feed_dict(self):
        return self._init_feed_dict


def default_init_op():
    return tf.group(
        tf.global_variables_initializer(),
        resources.initialize_resources(resources.shared_resources()))


def default_ready_op():
    return tf.concat([
        tf.report_uninitialized_variables(),
        resources.report_uninitialized_resources()], 0)


def default_ready_for_local_init_op():
    return tf.concat([
        tf.report_uninitialized_variables(
            tf.global_variables()),
        resources.report_uninitialized_resources(
            resources.shared_resources())], 0)


def default_local_init_op():
    return tf.group(
        tf.local_variables_initializer(),
        tf.tables_initializer(),
        resources.initialize_resources(resources.local_resources()))


def get_saver_or_default():
    collection_key = tf.GraphKeys.SAVERS
    savers = tf.get_collection(collection_key)
    if savers:
        if len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor."
                .format(collection_key))
        return savers[0]
    saver = tf.train.Saver(sharded=True, allow_empty=True)
    if saver is not None:
        tf.add_to_collection(collection_key, saver)
    return saver
