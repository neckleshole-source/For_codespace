"""FREUID Score metric.

Implements the competition's evaluation metric exactly as defined in README.md:

    g_audet  = 1 - AuDET
    g_apcer  = 1 - APCER@1%BPCER
    FREUID   = 1 - 2 * g_audet * g_apcer / (g_audet + g_apcer)

All components are bounded in [0, 1] and lower is better.

Definitions
-----------
- BPCER  (Bona-fide Presentation Classification Error Rate) = FPR
        = fraction of genuine (label=0) documents misclassified as fraud
        at a given decision threshold.
- APCER  (Attack Presentation Classification Error Rate)      = FNR
        = fraction of fraudulent (label=1) documents missed at a
        given decision threshold.
- AuDET  = area under the DET curve, i.e. the FNR-vs-FPR curve,
        integrated from FPR=0 to FPR=1 via the trapezoidal rule.
- APCER@1%BPCER = the APCER achieved when the threshold is set so
        BPCER equals 1% (FPR=0.01), linearly interpolated.

Higher fraud scores mean more likely fraud, so positive class is label=1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def audet(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the DET (FNR vs FPR) curve via trapezoidal integration.

    Returns a value in [0, 1]. 0 is perfect (no errors), 1 is worst.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    return float(np.trapezoid(fnr, fpr))


def apcer_at_bpcer(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bpcer_target: float = 0.01,
) -> float:
    """APCER (FNR) when the threshold is set so BPCER equals ``bpcer_target``.

    Linearly interpolates between adjacent ROC points when the exact
    target FPR is not present. Returns 1.0 if FPR never reaches the
    target (every threshold still rejects too many genuine docs) and
    0.0 if FPR is at or below the target from the very first point.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    # If the lowest FPR already meets the target, FNR at that point is the answer.
    if fpr[0] >= bpcer_target:
        return float(fnr[0])

    # Find the first index where FPR >= target and interpolate.
    for i in range(1, len(fpr)):
        if fpr[i] >= bpcer_target:
            fpr_lo, fpr_hi = fpr[i - 1], fpr[i]
            fnr_lo, fnr_hi = fnr[i - 1], fnr[i]
            if fpr_hi == fpr_lo:
                return float(fnr_hi)
            # Linear interpolation in (FPR, FNR) space.
            frac = (bpcer_target - fpr_lo) / (fpr_hi - fpr_lo)
            return float(fnr_lo + frac * (fnr_hi - fnr_lo))

    # FPR never reached the target — every threshold is too lenient on the
    # genuine class, so the strictest one (largest threshold) is the best.
    return float(fnr[-1])


def freuid_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bpcer_target: float = 0.01,
) -> float:
    """Combined FREUID Score for binary fraud detection.

    Args:
        y_true: ground-truth labels, 0 = bona-fide, 1 = fraud.
        y_score: predicted fraud scores, any real numbers, monotonically
                 comparable to each other (probabilities in [0,1] are typical
                 but not required by the formula).
        bpcer_target: BPCER operating point for the APCER component; the
                      competition uses 1% (0.01).

    Returns:
        Scalar in [0, 1]. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, got {y_true.shape} vs {y_score.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if not np.any(y_true == 0) or not np.any(y_true == 1):
        raise ValueError("y_true must contain both classes (0 and 1)")

    a = audet(y_true, y_score)
    p = apcer_at_bpcer(y_true, y_score, bpcer_target=bpcer_target)

    g_audet = 1.0 - a
    g_apcer = 1.0 - p

    # Harmonic mean of the two goodness scores, mapped back to a "badness"
    # score by 1 - HM. Guard the degenerate case where both goodness
    # scores are exactly 0 (perfect on both axes).
    denom = g_audet + g_apcer
    if denom <= 0.0:
        return 0.0
    hm = 2.0 * g_audet * g_apcer / denom
    return float(1.0 - hm)


if __name__ == "__main__":  # pragma: no cover — small self-test
    rng = np.random.default_rng(0)

    # Perfect predictions.
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    print(f"perfect: FREUID={freuid_score(y, s):.4f}, AuDET={audet(y, s):.4f}, APCER@1%={apcer_at_bpcer(y, s):.4f}")
    assert freuid_score(y, s) == 0.0

    # Random predictions should be near 1.0 (worst).
    y = rng.integers(0, 2, size=1000)
    s = rng.uniform(0, 1, size=1000)
    print(f"random:  FREUID={freuid_score(y, s):.4f}, AuDET={audet(y, s):.4f}, APCER@1%={apcer_at_bpcer(y, s):.4f}")
    assert 0.5 < freuid_score(y, s) < 1.0

    # Reversed scores — anti-perfect, should also be near worst.
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    print(f"reverse: FREUID={freuid_score(y, s):.4f}, AuDET={audet(y, s):.4f}, APCER@1%={apcer_at_bpcer(y, s):.4f}")
    assert freuid_score(y, s) > 0.5

    print("metric self-test OK")
