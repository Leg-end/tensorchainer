from tensorlib.engine.base_layer import Layer, param_tracker

__all__ = ['Lambda']


class Lambda(Layer):
    @param_tracker()
    def __init__(self,
                 function,
                 arguments=None,
                 arg_funcs=None,
                 activation=None,
                 **kwargs):
        assert callable(function), 'function must be a callable instance'
        if 'name' not in kwargs:
            kwargs['name'] = function.__name__[1:-1] \
                if function.__name__ == '<lambda>' else function.__name__
        super(Lambda, self).__init__(activation=activation, **kwargs)
        self.function = function
        if arguments is not None:
            assert isinstance(arguments, dict), 'function arguments must be a dict'
            self.arguments = arguments
        else:
            self.arguments = {}
        if arg_funcs is not None:
            assert isinstance(arg_funcs, dict), \
                'arg_funcs must be a dict with arg name as key, func as value'
            assert all(callable(v) for v in arg_funcs.values()),\
                'all values in arg_funcs must be callable'
            self.arg_funcs = arg_funcs
        else:
            self.arg_funcs = {}
        self.built = True

    def forward(self, *inputs, **kwargs):
        arguments = self.arguments.copy()
        arguments.update(kwargs)
        extra_args = {key: func(*inputs)
                      for key, func in self.arg_funcs.items()}
        arguments.update(extra_args)
        outputs = self.function(*inputs, **arguments)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs
