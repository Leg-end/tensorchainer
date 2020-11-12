import json
import copy
from tensorlib.utils import to_list


class HParams(object):
    @staticmethod
    def _convert_to_num(value):
        new_value = None
        try:
            if value.lower() in ('nan', 'inf'):
                flag = False
            else:
                new_value = float(value)
                flag = '.' in value
        except ValueError:
            flag = False
        if not flag:
            if value.isdigit():
                new_value = int(value)
                flag = True
            else:
                flag = False
        if flag:
            return new_value
        else:
            return value

    def __init__(self, help_dict=None, **kwargs):
        self.name = kwargs.get('name', None) or getattr(self, 'name', self.__class__.__name__)
        self.__dict__.update(kwargs)
        self.help = help_dict or dict.fromkeys(self.__dict__.keys())

    def __repr__(self):
        format_strs = [self.name + ':']
        param_dict = self._flatten().copy()
        for key, value in param_dict.items():
            if isinstance(value, dict):
                format_str = [key + ':']
                for k, v in value.items():
                    format_str.append("\t{} : {}".format(k, repr(v)))
                format_str = '\n'.join(format_str)
                format_strs.append(format_str)
            else:
                format_strs[0] += "\n\t{}: {}".format(key, repr(value))
        return '='*50 + '\n' + ('\n' + '-' * 30 + '\n').join(format_strs)

    def get_config(self):
        from tensorlib import saving
        config = self.__dict__.copy()
        config.pop('help')
        return saving.dump_dict(config)

    def del_hparam(self, name):
        if name in self.__dict__:
            self.__dict__.pop(name)
        else:
            print("There is no parameter named %s" % name)

    def update_hparam(self, values):
        assert isinstance(values, dict)
        self.__dict__.update(values)

    def set_hparam(self, name, value):
        if name in self.__dict__:
            if self.__dict__[name] is None:
                self.__dict__[name] = value
            else:
                tn = type(value)
                to = type(self.__dict__[name])
                if tn == to:
                    self.__dict__[name] = value
                else:
                    raise TypeError("Excepting value with type %s, but receive %s" % (to, tn))
        else:
            print("There is no parameter named %s" % name)

    def get_hparam(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return default

    def parse(self, values):
        if type(values) is not str:
            raise TypeError("Excepting argument with type 'str', but receive %s" % type(values))
        items = values.split(',')
        values_dict = dict()
        for item in items:
            (key, value) = item.split('=')
            values_dict[key] = self._convert_to_num(value)
        self.__dict__.update(**values_dict)
        return self

    def _flatten(self):
        state_dict = self.__dict__.copy()
        state_dict.pop('help')
        for key, value in state_dict.items():
            if isinstance(value, HParams):
                state_dict[key] = value._flatten()
        return state_dict

    def _flatten_dict(self):
        state_dict = self.__dict__.copy()
        state_dict.pop('help')
        remove_keys = []
        for key, value in state_dict.items():
            if isinstance(value, HParams):
                tmp = value._flatten_dict()
                for k, v in tmp.items():
                    state_dict[key + '_' + k] = v
                    self.help[key + '_' + k] = v.help[k]
                remove_keys.append(key)
        for key in remove_keys:
            state_dict.pop(key)
        return state_dict

    def to_json(self, indent=2, separators=None, sort_keys=False):
        from tensorlib import saving
        return json.dumps(saving.dump_object(self), indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def to_json_file(self, path, indent=2, separators=None, sort_keys=False):
        from tensorlib import saving
        json.dump(saving.dump_object(self), open(path, 'w'), indent=indent,
                  separators=separators, sort_keys=sort_keys)

    def values(self, ignore_params=None):
        params = self._flatten()
        if ignore_params is not None:
            ignore_params = to_list(ignore_params)
            if any([not isinstance(param, str) for param in ignore_params]):
                raise TypeError("Value in `ignore_params` must be str")
            for param in ignore_params:
                params.pop(param)
        return params

    def to_arg_parser(self, min_prefix=3):
        import argparse
        parser = argparse.ArgumentParser()
        used_flags = {}
        for key, value in self._flatten_dict().items():
            if '_' in key:
                flag = ''.join([v[0] for v in key.split('_')])
            else:
                flag = key[:min(min_prefix, len(key))]
            if flag not in used_flags:
                used_flags[flag] = 0
            else:
                used_flags[flag] += 1
                flag += str(used_flags[flag])
            flag = '-' + flag.lower()
            name = '--' + key.lower()
            parser.add_argument(flag, name, type=type(value), default=value,
                                help=self.help[key] or 'usage: <{}> or <{}>'.format(flag, name))
        return parser.parse_args()

    def copy(self):
        return copy.deepcopy(self)


class DataConfig(HParams):
    def __init__(self, help_dict=None, **kwargs):
        self.data_dir = ''
        self.feature_dir_name = ''
        self.label_dir_name = ''
        self.annotation = ''
        self.batch_size = 32
        self.padded_batch = False
        self.padded_values = None
        self.buffer = 50
        self.prefetch = 10
        self.num_parallel_calls = 4
        self.num_class = 7
        self.input_shape = None
        self.label_shape = None
        self.input_shapes = None
        self.static_shapes = None
        self.input_types = None
        self.shuffle = False
        self.repeats = 1
        self.class_names = None
        super(DataConfig, self).__init__(help_dict=help_dict, **kwargs)


class EnvironmentConfig(HParams):
    def __init__(self, help_dict=None, **kwargs):
        self.random_seed = 666
        self.CUDA_VISIBLE_DEVICES = '0'
        self.per_process_gpu_memory_fraction = 0.4
        self.intra_op_parallelism_threads = 2
        self.inter_op_parallelism_threads = 8
        self.num_parallel_calls = 4  # tf.dataset.experimental.AUTOTUNE
        self.prefetch_size = 8  # tf.dataset.experimental.AUTOTUNE
        self.allow_growth = True
        super(EnvironmentConfig, self).__init__(help_dict=help_dict, **kwargs)


class RunConfig(HParams):
    def __init__(self, help_dict=None, **kwargs):
        self.model_name = 'model'
        self.pretrained_path = ''
        self.model_dir = ''
        self.store_dir = ''
        self.root_dir = './'
        self.test_dir = ''
        self.class_names = None
        self.steps = 70000
        self.batch_size = 32
        self.input_shape = None
        self.label_shape = None
        self.log_frequency = 1
        self.save_checkpoints_steps = 500
        self.keep_checkpoint_max = 2
        self.keep_checkpoint_every_n_hours = 6
        self.save_summary_steps = 500
        self.optimizer = 'sgd'
        self.learning_rate = 0.01
        self.scheduler = ''
        self.decay_steps = 500
        self.decay_rate = 0.85
        self.staircase = True
        self.boundaries = None
        self.lr_values = None
        self.from_ckpt = False
        super(RunConfig, self).__init__(help_dict=help_dict, **kwargs)


if __name__ == '__main__':
    # config = BaseConfig(model_name='resnet50')
    # config.to_json_file('./config.txt')
    import tensorflow as tf
    config = DataConfig(annotation=r'D:\zhangzk\data\pa100kanno\new_anno.txt',
                        feature_dir_name=r'D:\pedestrian\pa100k_2',
                        label_dir_name='',
                        input_types=(tf.float32, tf.int64),
                        input_shapes=((None, None, 3), (8,)),
                        num_class=8,
                        batch_size=1)
    args = config.to_arg_parser()
    print(args)
