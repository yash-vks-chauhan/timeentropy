import numpy as np

from timeentropy.signal.features import log_magnitude_stft


def test_stft_is_frequency_by_time_and_standardized() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=3_680).astype(np.float32)
    feature = log_magnitude_stft(signal)
    assert feature.shape == (257, 21)
    assert abs(float(feature.mean())) < 1.0e-5
    assert abs(float(feature.std()) - 1.0) < 1.0e-5
