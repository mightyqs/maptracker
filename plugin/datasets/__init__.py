import logging

from .pipelines import *
from .nusc_dataset import NuscDataset


try:
    from .argo_dataset import AV2Dataset
except ImportError as exc:
    AV2Dataset = None
    logging.getLogger(__name__).warning(
        'AV2 dataset support is unavailable: %s', exc
    )
except TypeError as exc:
    if 'Type subscription requires python >= 3.9' not in str(exc):
        raise
    AV2Dataset = None
    logging.getLogger(__name__).warning(
        'AV2 dataset support is unavailable on this Python runtime: %s', exc
    )
