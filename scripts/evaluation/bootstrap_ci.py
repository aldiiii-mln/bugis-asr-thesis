"""
Bootstrap confidence intervals for WER/CER (Section 4.3 / 5). Applied
consistently to both metrics — no double standard between them.

Especially important for small test sets (Authentic-Only, N=5-ish per fold)
where a point-estimate WER can be misleadingly precise-looking.
"""

import numpy as np
from compute_wer_cer import compute_wer, compute_cer


def bootstrap_ci(
    references: list[str],
    hypotheses: list[str],
    metric_fn=compute_wer,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    assert len(references) == len(hypotheses)
    n = len(references)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(references, hypotheses)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        r_sample = [references[i] for i in idx]
        h_sample = [hypotheses[i] for i in idx]
        scores.append(metric_fn(r_sample, h_sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return {
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="tab-separated file_id\\treference\\thypothesis")
    parser.add_argument("--metric", choices=["wer", "cer"], default="wer")
    args = parser.parse_args()

    refs, hyps = [], []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        _, ref, hyp = line.split("\t")
        refs.append(ref)
        hyps.append(hyp)

    fn = compute_wer if args.metric == "wer" else compute_cer
    result = bootstrap_ci(refs, hyps, metric_fn=fn)
    print(json.dumps(result, indent=2))
