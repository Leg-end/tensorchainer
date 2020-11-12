from tensorlib.engine.base_layer import LayerList
from tensorlib.utils.generic_util import unpack_singleton, to_list


class Sequential(LayerList):

    def flatten(self):
        ret = Sequential()
        for layer in self:
            if isinstance(layer, Sequential):
                ret.extend(layer.flatten())
            else:
                ret.append(layer)
        return ret

    def forward(self, *inputs):
        if len(self) == 0:
            raise RuntimeError("Can not run on empty layer")
        for layer in self:
            inputs = to_list(layer(*inputs))
        return unpack_singleton(inputs)
