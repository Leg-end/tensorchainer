import tensorlib as lib


@lib.engine.add_arg_scope
def Dense(units,
          kernel_initializer='truncated_normal',
          kernel_regularizer=None,
          bias_initializer='zeros',
          bias_regularizer=None,
          activation=None,
          normalizer=None,
          normalizer_params=None,
          trainable=False,
          name=None):
    dense = lib.layers.Dense(
        units=units,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activation=None if normalizer else activation,
        use_bias=not normalizer and bias_initializer,
        trainable=trainable,
        name=name)
    if normalizer is not None:
        assert issubclass(normalizer, lib.Layer)
        normalizer_params = normalizer_params or {}
        bn = normalizer(name=dense.name + '/batch_norm',
                        activation=activation,
                        **normalizer_params)
        return lib.Sequential(dense, bn, name='')
    return dense
