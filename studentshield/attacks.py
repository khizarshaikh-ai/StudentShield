
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from .core import make_pipeline, per_record_loss


def run_shadow_lira(
    X,
    y,
    task,
    model_name,
    target_model,
    target_train_idx,
    audit_idx,
    n_shadows=30,
    shadow_fraction=None,
    seed=123,
    threshold=0.0,
):
    """
    Gaussian LiRA-style membership inference.

    Positive scores indicate evidence in favor of membership.

    The audit set must contain both:
        1. Records used to train the target model.
        2. Records not used to train the target model.

    shadow_fraction:
        If None, derive the fraction from the target training
        size relative to the experiment pool.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------

    if task not in ("Regression", "Grade bands"):
        raise ValueError(f"Unsupported task: {task}")

    if not isinstance(n_shadows, (int, np.integer)):
        raise ValueError("n_shadows must be an integer.")

    if n_shadows < 4:
        raise ValueError(
            "Use at least 4 shadow models. "
            "30 or more are recommended for more stable estimates."
        )

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    target_train_idx = np.asarray(
        target_train_idx, dtype=int
    )

    audit_idx = np.asarray(
        audit_idx, dtype=int
    )

    if len(target_train_idx) == 0:
        raise ValueError("Target training set is empty.")

    if len(audit_idx) < 2:
        raise ValueError("Not enough audit records.")

    if len(np.unique(target_train_idx)) != len(target_train_idx):
        raise ValueError(
            "Duplicate target training indices detected."
        )

    if len(np.unique(audit_idx)) != len(audit_idx):
        raise ValueError(
            "Duplicate audit records detected."
        )

    if (
        np.any(target_train_idx < 0)
        or np.any(target_train_idx >= len(X))
        or np.any(audit_idx < 0)
        or np.any(audit_idx >= len(X))
    ):
        raise ValueError(
            "Training or audit indices are outside dataset bounds."
        )

    # --------------------------------------------------
    # MEMBERSHIP LABELS
    # --------------------------------------------------

    target_train_set = set(
        target_train_idx.tolist()
    )

    target_member = np.array([
        int(i in target_train_set)
        for i in audit_idx
    ])

    if len(np.unique(target_member)) != 2:
        raise ValueError(
            "Audit set must contain both members and non-members. "
            "Select some records from target_train_idx and some "
            "from the target holdout set."
        )

    # The pool includes target training records and
    # the selected audit records.
    experiment_idx = np.unique(
        np.concatenate([
            target_train_idx,
            audit_idx
        ])
    )

    if len(experiment_idx) < 3:
        raise ValueError(
            "The experiment pool needs at least 3 records."
        )

    X_a = X.iloc[audit_idx]
    y_a = y.iloc[audit_idx]

    # --------------------------------------------------
    # SHADOW FRACTION
    # --------------------------------------------------

    if shadow_fraction is None:
        shadow_fraction = (
            len(target_train_idx) / len(experiment_idx)
        )

    if not 0 < shadow_fraction < 1:
        raise ValueError(
            "shadow_fraction must be between 0 and 1."
        )

    n_train = int(
        round(len(experiment_idx) * shadow_fraction)
    )

    n_train = max(
        1,
        min(len(experiment_idx) - 1, n_train)
    )

    # --------------------------------------------------
    # SHADOW LOSS STORAGE
    # --------------------------------------------------

    in_scores = [
        [] for _ in range(len(audit_idx))
    ]

    out_scores = [
        [] for _ in range(len(audit_idx))
    ]

    # --------------------------------------------------
    # TRAIN SHADOW MODELS
    # --------------------------------------------------

    for s in range(n_shadows):

        if task == "Grade bands":

            chosen_parts = []

            pool_labels = y.iloc[
                experiment_idx
            ].to_numpy()

            for cls in pd.unique(pool_labels):

                class_indices = experiment_idx[
                    pool_labels == cls
                ]

                if len(class_indices) < 2:
                    raise ValueError(
                        f"Grade band '{cls}' has fewer than "
                        "2 records in the shadow pool. "
                        "IN/OUT loss estimation requires "
                        "records from both groups."
                    )

                n_class = int(round(
                    len(class_indices) * shadow_fraction
                ))

                n_class = max(
                    1,
                    min(len(class_indices) - 1, n_class)
                )

                selected = rng.choice(
                    class_indices,
                    size=n_class,
                    replace=False
                )

                chosen_parts.extend(
                    selected.tolist()
                )

            chosen = np.asarray(
                chosen_parts,
                dtype=int
            )

        else:

            chosen = rng.choice(
                experiment_idx,
                size=n_train,
                replace=False
            )

        chosen_set = set(
            chosen.tolist()
        )

        shadow = make_pipeline(
            X.iloc[chosen],
            task,
            model_name,
            seed + s + 1
        )

        shadow.fit(
            X.iloc[chosen],
            y.iloc[chosen]
        )

        losses = np.asarray(
            per_record_loss(
                shadow,
                X_a,
                y_a,
                task
            ),
            dtype=float
        )

        if len(losses) != len(audit_idx):
            raise RuntimeError(
                "Shadow model returned an unexpected "
                "number of record losses."
            )

        if not np.all(np.isfinite(losses)):
            raise RuntimeError(
                "Shadow model produced non-finite losses."
            )

        for j, original_idx in enumerate(audit_idx):

            if int(original_idx) in chosen_set:
                in_scores[j].append(
                    float(losses[j])
                )
            else:
                out_scores[j].append(
                    float(losses[j])
                )

    # --------------------------------------------------
    # TARGET MODEL LOSSES
    # --------------------------------------------------

    target_losses = np.asarray(
        per_record_loss(
            target_model,
            X_a,
            y_a,
            task
        ),
        dtype=float
    )

    if not np.all(np.isfinite(target_losses)):
        raise RuntimeError(
            "Target model produced non-finite losses."
        )

    # --------------------------------------------------
    # GAUSSIAN LIRA SCORES
    # --------------------------------------------------

    records = []

    for j, original_idx in enumerate(audit_idx):

        ins = np.asarray(
            in_scores[j],
            dtype=float
        )

        outs = np.asarray(
            out_scores[j],
            dtype=float
        )

        if len(ins) < 2 or len(outs) < 2:
            raise RuntimeError(
                f"Audit record {original_idx} has insufficient "
                f"shadow observations: IN={len(ins)}, "
                f"OUT={len(outs)}. Increase n_shadows or "
                "adjust shadow_fraction."
            )

        mu_in = float(ins.mean())
        mu_out = float(outs.mean())

        pooled = np.concatenate([
            ins,
            outs
        ])

        pooled_sd = float(
            np.std(pooled, ddof=1)
        )

        sd_floor = max(
            pooled_sd * 0.05,
            1e-3
        )

        sd_in = max(
            float(ins.std(ddof=1)),
            sd_floor
        )

        sd_out = max(
            float(outs.std(ddof=1)),
            sd_floor
        )

        log_likelihood_in = float(
            norm.logpdf(
                target_losses[j],
                loc=mu_in,
                scale=sd_in
            )
        )

        log_likelihood_out = float(
            norm.logpdf(
                target_losses[j],
                loc=mu_out,
                scale=sd_out
            )
        )

        score = (
            log_likelihood_in
            - log_likelihood_out
        )

        if not np.isfinite(score):
            raise RuntimeError(
                f"Non-finite LiRA score for record {original_idx}."
            )

        records.append({
            "row_id": int(original_idx),
            "actual_member": int(target_member[j]),
            "target_loss": float(target_losses[j]),
            "shadow_in_count": int(len(ins)),
            "shadow_out_count": int(len(outs)),
            "shadow_in_mean": mu_in,
            "shadow_out_mean": mu_out,
            "shadow_in_sd": sd_in,
            "shadow_out_sd": sd_out,
            "lira_log_lr": float(score),
        })

    result = pd.DataFrame(records)

    # --------------------------------------------------
    # MEMBERSHIP CLASSIFICATION
    # --------------------------------------------------

    result["predicted_member"] = (
        result["lira_log_lr"] > threshold
    ).astype(int)

    truth = result[
        "actual_member"
    ].to_numpy()

    pred = result[
        "predicted_member"
    ].to_numpy()

    scores = result[
        "lira_log_lr"
    ].to_numpy()

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        truth,
        pred,
        labels=[0, 1]
    ).ravel()

    tpr = (
        tp / (tp + fn)
        if (tp + fn) else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) else 0.0
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    metrics = {
        "roc_auc": float(
            roc_auc_score(truth, scores)
        ),

        "accuracy": float(
            accuracy_score(truth, pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(truth, pred)
        ),

        "precision": float(
            precision_score(
                truth,
                pred,
                zero_division=0
            )
        ),

        "recall": float(tpr),

        "f1": float(
            f1_score(
                truth,
                pred,
                zero_division=0
            )
        ),

        "tpr": float(tpr),
        "fpr": float(fpr),

        "attack_advantage": float(
            tpr - fpr
        ),

        "threshold": float(threshold),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ],

        "n_audit_records": int(
            len(audit_idx)
        ),

        "n_members_audited": int(
            (truth == 1).sum()
        ),

        "n_nonmembers_audited": int(
            (truth == 0).sum()
        ),

        "n_shadows": int(n_shadows),

        "shadow_fraction": float(
            shadow_fraction
        ),
    }

    return result, metrics
