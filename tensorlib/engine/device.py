from tensorflow.core.framework import device_attributes_pb2
from tensorflow.python import pywrap_tensorflow


def list_local_devices(session_config=None):
    def _convert(pb_str):
        m = device_attributes_pb2.DeviceAttributes()
        m.ParseFromString(pb_str)
        return m

    return [_convert(s) for s in pywrap_tensorflow.list_devices(
        session_config=session_config)]
