import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from studentshield.core import prepare_xy, make_pipeline, per_record_loss
from studentshield.attacks import run_shadow_lira

def test_student_shield_pipeline():
    print("--- 1. Testing Data Pipeline & Training ---")
    
    # Dataset size ko 1000 kar diya hai taake robust shadow distribution ban sake
    np.random.seed(42)
    n_samples = 1000
    dummy_df = pd.DataFrame({
        "studytime": np.random.randint(1, 5, n_samples),
        "failures": np.random.choice([0, 1, 2], n_samples),
        "absences": np.random.randint(0, 20, n_samples),
        "G3": np.random.randint(0, 20, n_samples) # Target column
    })
    
    target = "G3"
    task = "Regression"
    features = ["studytime", "failures", "absences"]
    
    # X, y preparation check
    X, y, used_features = prepare_xy(dummy_df, target, task, features)
    assert len(X) == n_samples, "Row count mismatch in prepare_xy"
    print(f"✓ Data prepared successfully. Features used: {used_features}")
    
    # Train-test split
    idx = np.arange(len(X))
    train_idx, holdout_idx = train_test_split(idx, test_size=0.3, random_state=42)
    
    # Model pipeline check
    model = make_pipeline(X.iloc[train_idx], task, "Random Forest", 42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    print("✓ Target model trained successfully.")
    
    # Per-record loss check
    audit_idx = np.concatenate([train_idx[:20], holdout_idx[:20]])
    losses = per_record_loss(model, X.iloc[audit_idx], y.iloc[audit_idx], task)
    assert len(losses) == len(audit_idx), "Loss calculation count mismatch"
    print(f"✓ Per-record loss calculated. Sample losses: {losses[:3]}")
    
    print("\n--- 2. Testing Shadow Model & LiRA Attack ---")
    # n_shadows ko 30 kar diya hai taake sufficient IN/OUT observations mil sakein
    results, metrics = run_shadow_lira(
        X, y, task, "Random Forest",
        model, train_idx, audit_idx,
        n_shadows=30, shadow_fraction=0.5, seed=42
    )
    
    assert "roc_auc" in metrics, "ROC-AUC metric missing from LiRA output"
    assert not results.empty, "Results dataframe is empty"
    
    print(f"✓ LiRA Attack test completed successfully!")
    print(f"Metrics Output: {metrics}")

if __name__ == "__main__":
    test_student_shield_pipeline()