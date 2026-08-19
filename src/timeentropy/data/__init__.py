"""Dataset manifests, deterministic partitions, and audio runtime helpers."""

from timeentropy.data.audio import LoadedAudio, load_audio
from timeentropy.data.augmentation import (
    AugmentationConfig,
    augment_waveform,
    augmentation_rng,
    mask_spectrogram,
)
from timeentropy.data.dataset import PrefixFeatureDataset, collate_prefix_features
from timeentropy.data.raw import RawPrefixDataset, collate_raw_prefix_batch
from timeentropy.data.sampling import class_balanced_weights, make_class_balanced_sampler
from timeentropy.data.validation import (
    assign_validation_partitions,
    attach_validation_partitions,
    validation_partition,
)

__all__ = [
    "AugmentationConfig",
    "LoadedAudio",
    "PrefixFeatureDataset",
    "RawPrefixDataset",
    "assign_validation_partitions",
    "attach_validation_partitions",
    "augment_waveform",
    "augmentation_rng",
    "class_balanced_weights",
    "collate_prefix_features",
    "collate_raw_prefix_batch",
    "load_audio",
    "make_class_balanced_sampler",
    "mask_spectrogram",
    "validation_partition",
]
