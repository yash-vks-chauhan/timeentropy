import json

import numpy as np
import pandas as pd

from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.entropy import normalized_binary_entropy
from timeentropy.provenance import sha256_file
from timeentropy.reporting.pipeline import generate_report_bundle


def _valid_ledgers() -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows = []
    decision_rows = []
    for index, label in enumerate((0, 0, 1, 1)):
        run_id = f"run-{index}"
        sample_id = f"sample-{index}"
        logits = np.asarray([[2.0, -1.0]]) if label == 0 else np.asarray([[-1.0, 2.0]])
        probabilities = probabilities_from_logits(logits)[0]
        for prefix_ms in (150, 200, 250, 300):
            prefix_rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "speaker_id": f"speaker-{index}",
                    "fold": index,
                    "seed": 20260819,
                    "model_id": "transformer",
                    "split_role": "test",
                    "onset_variant": "causal_fixed",
                    "pre_roll_ms": 30,
                    "prefix_ms": prefix_ms,
                    "label_id": label,
                    "logit_control": logits[0, 0],
                    "logit_dysarthric": logits[0, 1],
                    "prob_control": probabilities[0],
                    "prob_dysarthric": probabilities[1],
                    "predicted_label": label,
                    "normalized_entropy": float(normalized_binary_entropy(probabilities)),
                }
            )
        decision_rows.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "policy_id": "complete",
                "label_id": label,
                "decision_time_ms": 200,
                "predicted_label": label,
                "forced_fallback": False,
                "accepted_at_endpoint": False,
            }
        )
    return pd.DataFrame(prefix_rows), pd.DataFrame(decision_rows)


def test_report_bundle_writes_csv_json_and_provenance(tmp_path) -> None:
    prefix, decisions = _valid_ledgers()
    prefix_path = tmp_path / "prefix.csv"
    decision_path = tmp_path / "decisions.csv"
    config_path = tmp_path / "frozen.yaml"
    output_path = tmp_path / "reports"
    prefix.to_csv(prefix_path, index=False)
    decisions.to_csv(decision_path, index=False)
    config_path.write_text("protocol: frozen\n", encoding="utf-8")

    manifest = generate_report_bundle(
        prefix_path,
        decision_path,
        output_path,
        config_path=config_path,
        bootstrap_replicates=20,
        include_model_comparisons=False,
    )
    assert (output_path / "pooled_metrics.csv").is_file()
    assert (output_path / "pooled_metrics.json").is_file()
    assert (output_path / "calibration_metrics.csv").is_file()
    written = json.loads((output_path / "report_manifest.json").read_text(encoding="utf-8"))
    assert written == manifest
    assert written["config_sha256"] == sha256_file(config_path)
    assert written["prefix_ledger_sha256"] == sha256_file(prefix_path)
