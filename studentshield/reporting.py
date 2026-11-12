
from datetime import datetime


def make_report(
    dataset_name, target, task, model_name,
    metrics, n_rows, n_features, n_shadows
):
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

    return f"""# StudentShield Research Report

- Generated: {datetime.now().isoformat(timespec='seconds')}
- Dataset: {dataset_name}
- Target: {target}
- Task: {task}
- Target model: {model_name}
- Usable dataset rows: {n_rows}
- Input features: {n_features}
- Shadow models: {n_shadows}

## Membership Inference Attack

| Metric | Result |
|---|---:|
| ROC-AUC | {metrics.get('roc_auc', float('nan')):.4f} |
| Accuracy | {metrics.get('accuracy', float('nan')):.4f} |
| Balanced Accuracy | {metrics.get('balanced_accuracy', float('nan')):.4f} |
| Precision | {metrics.get('precision', float('nan')):.4f} |
| Recall / TPR | {metrics.get('tpr', float('nan')):.4f} |
| FPR | {metrics.get('fpr', float('nan')):.4f} |
| F1 | {metrics.get('f1', float('nan')):.4f} |
| Attack Advantage | {metrics.get('attack_advantage', float('nan')):.4f} |

## Audit Sample

- Total audited: {metrics.get('n_audit_records', 0)}
- Members: {metrics.get('n_members_audited', 0)}
- Non-members: {metrics.get('n_nonmembers_audited', 0)}
- Decision threshold: {metrics.get('threshold', 0.0)}

## Confusion Matrix

Rows: actual [non-member, member].
Columns: predicted [non-member, member].

| | Predicted Non-member | Predicted Member |
|---|---:|---:|
| Actual Non-member | {matrix[0][0]} | {matrix[0][1]} |
| Actual Member | {matrix[1][0]} | {matrix[1][1]} |

## Interpretation

ROC-AUC measures ranking discrimination between members and non-members.
Accuracy and balanced accuracy measure classification performance.
Attack advantage is TPR minus FPR.

These metrics describe this specific experiment, not a universal
privacy guarantee.

The implementation uses Gaussian per-record loss distributions
and is a LiRA-style educational approximation.

Results should be reproduced with independent random seeds,
adequate sample sizes, and confidence intervals before drawing
research conclusions.

Small datasets and limited shadow models can produce unstable
membership inference estimates.
"""
