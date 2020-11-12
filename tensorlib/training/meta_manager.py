from datetime import datetime
import json
import os
from tensorlib import saving


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


class RunMetaManager:

    @staticmethod
    def register_meta(root_dir: str, tag, *configs):
        import json
        mkdir(root_dir)
        history_path = os.path.join(root_dir, 'history.json')
        store_dir = os.path.join(root_dir, tag)
        mkdir(store_dir)
        model_dir = os.path.join(store_dir, 'model')
        mkdir(model_dir)
        test_dir = os.path.join(store_dir, 'test')
        mkdir(test_dir)
        configs[0].update_hparam(dict(store_dir=store_dir,
                                      model_dir=model_dir,
                                      test_dir=test_dir))
        paths = [os.path.join(store_dir, cfg.name + '.json') for cfg in configs]
        [cfg.to_json_file(path) for cfg, path in zip(configs, paths)]
        if os.path.exists(history_path):
            history = json.load(open(history_path))
            # Remove useless history
            keys = list(history.keys())
            for key in keys:
                if not os.path.exists(history[key]['paths'][0]):
                    print('Clear useless {}'.format(key))
                    history.pop(key)
        else:
            history = dict()
        # Update history
        history[tag] = {'paths': paths, 'timestamp': str(datetime.now())}
        json.dump(history, open(history_path, 'w'), indent=2)
        print("Successfully updating history {} in to {})".format(
            tag, history_path))

    @staticmethod
    def get_meta(tag, root_dir: str):
        history_path = os.path.join(root_dir, 'history.json')
        if not os.path.exists(history_path):
            raise ValueError("Can not find history file, you may forget "
                             "to record config first.")
        history = json.load(open(history_path))
        if tag not in history:
            raise ValueError("Can not find {}'s config in history records,"
                             "you may forget to record {}'s config first.".format(
                              tag, tag))
        print('==>Read configs from:\n' + '\n\t'.join(history[tag]['paths']))
        return tuple(saving.from_json_file(path) for path in history[tag]['paths'])
