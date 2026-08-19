from pathlib import Path

from timeentropy.splits import load_folds


def test_every_speaker_is_tested_once() -> None:
    folds = load_folds(Path("configs/folds_v1.yaml"))
    tested = [speaker for fold in folds for speaker in fold.test]
    assert len(tested) == len(set(tested)) == 15
    for fold in folds:
        assert not (set(fold.train) & set(fold.validation))
        assert not (set(fold.train) & set(fold.test))
        assert not (set(fold.validation) & set(fold.test))
