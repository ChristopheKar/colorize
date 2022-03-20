from .base import ZhangBase
from .eccv16 import ZhangECCV16
from .siggraph17 import ZhangSIGGRAPH17


AVAILABLE_MODELS = {
    'zhang-eccv16-caffe': {
        'model': ZhangECCV16,
        'kwargs': {
            'caffe': True
        }
    },
    'zhang-eccv16': {
        'model': ZhangECCV16,
        'kwargs': {
            'caffe': False
        }
    },
    'zhang-siggraph17': {
        'model': ZhangSIGGRAPH17,
        'kwargs': {}
    }
}
