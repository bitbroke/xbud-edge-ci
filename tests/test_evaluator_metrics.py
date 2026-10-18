import pytest
from benchmarks import evaluation_metrics as metrics


def test_shared_case_and_whitespace_policy():
    result = metrics.compute_wer_cer_metrics(["Hello world", " A  test "], ["HELLO WORLD", "a test"])
    assert result["corpus_wer"] == result["mean_clip_wer"] == 0
    assert result["corpus_cer"] == result["mean_clip_cer"] == 0


def test_corpus_is_word_weighted_not_clip_mean():
    result = metrics.compute_wer_cer_metrics(["one", "two three four"], ["", "two three four"])
    assert result["corpus_wer"] == 0.25
    assert result["mean_clip_wer"] == 0.5


def test_scoring_error_is_not_silenced(monkeypatch):
    def broken(*args):
        raise RuntimeError("metric failure")
    monkeypatch.setattr(metrics, "process_words", broken)
    with pytest.raises(RuntimeError, match="metric failure"):
        metrics.compute_wer_cer_metrics(["test"], ["test"])


@pytest.mark.parametrize("refs,hyps", [([], []), (["x"], []), ([""], ["noise"])])
def test_invalid_metric_inputs(refs, hyps):
    with pytest.raises(ValueError):
        metrics.compute_wer_cer_metrics(refs, hyps)


def test_exact_date_catches_same_time_wrong_day():
    with pytest.raises(AssertionError):
        metrics.assert_exact_utc("2026-10-18T03:30:00Z", "2026-10-19T03:30:00Z")


def test_timestamp_offsets_and_naive_rejection():
    metrics.assert_exact_utc("2026-10-19T09:00:00+05:30", "2026-10-19T03:30:00Z")
    with pytest.raises(ValueError):
        metrics.assert_exact_utc("2026-10-19T03:30:00", "2026-10-19T03:30:00Z")


def test_manifest_has_full_utc_oracles():
    import json
    from pathlib import Path
    manifest = json.loads((Path(__file__).parents[1] / "datasets/arm64_eval_corpus/commands_manifest.json").read_text())
    for session in manifest:
        outcome = session["expected_outcome"]
        if "verified_reminder" in outcome:
            value = outcome["verified_reminder"]["due_time_utc"]
            metrics.assert_exact_utc(value, value)
        if "verified_event" in outcome:
            for field in ("start_time_utc", "end_time_utc"):
                value = outcome["verified_event"][field]
                metrics.assert_exact_utc(value, value)
