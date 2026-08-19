from timeentropy.data.recover import _member_suffix, speaker_group


def test_speaker_group_prefers_control_prefixes() -> None:
    assert speaker_group("F01") == "F"
    assert speaker_group("FC01") == "FC"
    assert speaker_group("M01") == "M"
    assert speaker_group("MC01") == "MC"


def test_member_suffix_uses_index_relative_layout() -> None:
    value = _member_suffix("TORGO/F/F01/Session1/wav_headMic/0001.wav")
    assert value == "F01/Session1/wav_headMic/0001.wav"
