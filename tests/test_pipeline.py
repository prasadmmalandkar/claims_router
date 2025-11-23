import pandas as pd
from src.train_claims_agent import main

def test_training_runs(tmp_path):
    # Create dummy CSV
    df = pd.DataFrame({
        "Claims No": ["CLM001"],
        "Claim Status": ["Approved"],
        "Claim Charge Amount": [100],
        "Diagnosis": ["I10"]
    })
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)

    # Run training
    main(str(csv_path), target_col="Claim Status", outdir=str(tmp_path))
    assert (tmp_path / "model.pkl").exists()