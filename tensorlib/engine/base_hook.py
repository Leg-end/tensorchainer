import tensorlib


class Hook(object):

    name = 'Hook'

    def __enter__(self):
        hooks = tensorlib._get_hooks()
        if self.name in hooks:
            raise KeyError('Hook %s already exists' % self.name)
        hooks[self.name] = self
        self.added(None)
        return self

    def __exit__(self, *_):
        hooks = tensorlib._get_hooks()
        hooks[self.name].deleted(None)
        del hooks[self.name]

    def added(self, layer) -> None:
        pass

    def deleted(self, layer) -> None:
        pass

    def before_forward(self, layer, args: (list, tuple), kwargs: dict) -> None:
        pass

    def after_forward(self, layer, outputs, args: (list, tuple), kwargs: dict) -> None:
        pass
