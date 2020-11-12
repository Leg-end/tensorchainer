from tensorflow import train
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops as fops
from tensorlib.engine import base_lib as F


class Optimizer(object):

    def __init__(self, optimizer: train.Optimizer, global_step):
        self.optimizer = optimizer
        self.global_step = global_step

    def get_updates(self, loss, params):
        grads = self.optimizer.compute_gradients(loss, params)
        opt_update = self.optimizer.apply_gradients(
            grads, global_step=self.global_step)
        return [opt_update]


class MultiLROptimizer(Optimizer):

    def __init__(self, optimizer: train.Optimizer, global_step, lr_multiplier):
        super(MultiLROptimizer, self).__init__(optimizer, global_step)
        if not isinstance(lr_multiplier, dict):
            raise ValueError("`lr_multiplier` must has type dict,"
                             " but received: ", str(type(lr_multiplier)))
        if len(lr_multiplier) == 0:
            raise ValueError("`lr_multiplier` can not be empty")
        self.lr_multiplier = lr_multiplier

    def get_updates(self, loss, params):
        multiplied_grads_and_vars = []

        def _get_multiplier(name):
            for key, value in self.lr_multiplier.items():
                if key in name:
                    return self.lr_multiplier[key]
            return None

        grads_and_vars = self.optimizer.compute_gradients(loss, params)
        base_lr = getattr(self.optimizer, '_lr')
        none_counts = 0
        for grad, var in grads_and_vars:
            multiplier = _get_multiplier(var.op.name)
            if grad is None:
                none_counts += 1
            if multiplier is not None:
                if grad is None:
                    raise ValueError('Requested multiple of `None` gradient.')
                if callable(multiplier):
                    lr = multiplier(self.global_step, base_lr)
                elif not F.is_tensor(multiplier):
                    lr = array_ops.constant(multiplier) * base_lr
                else:
                    lr = multiplier * base_lr
                if isinstance(grad, fops.IndexedSlices):
                    tmp = grad.values * lr
                    grad = fops.IndexedSlices(
                        tmp, grad.indices, grad.dense_shape)
                else:
                    grad *= lr
            multiplied_grads_and_vars.append((grad, var))
        if none_counts == len(multiplied_grads_and_vars):
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))
        opt_update = self.optimizer.apply_gradients(
            multiplied_grads_and_vars, global_step=self.global_step)
        return [opt_update]
