import argparse, os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# Column descriptions (metadata)
COLUMN_DESCRIPTIONS = {
    "Claims No": "Unique identifier for each claim submitted to the payer.",
    "Claim Line": "Line item within the claim, representing a specific service or procedure.",
    "Member ID": "Identifier for the patient/member receiving care.",
    "Provider ID": "Identifier for the healthcare provider submitting the claim.",
    "Line of Business ID": "Insurance product line (commercial, Medicare, Medicaid).",
    "Revenue Code": "Department or type of service billed.",
    "Service Code": "Internal code representing the type of service rendered.",
    "Procedure Code": "CPT/HCPCS code describing the medical procedure performed.",
    "Procedure Description": "Text description of the procedure code.",
    "Diagnosis": "ICD-10 code representing the patient’s condition.",
    "Claim Charge Amount": "Total amount billed for the claim line.",
    "Denial Reason Code": "Code explaining why a claim was denied.",
    "Price Index": "Relative pricing factor for benchmarking.",
    "In / Out Network": "Indicates whether provider is in-network or out-of-network.",
    "Reference Index": "Internal reference number for tracking.",
    "Subscriber Payment Amount": "Amount paid by the subscriber.",
    "Insurance Eligible Amount": "Portion of claim eligible for insurance reimbursement.",
    "Group Index": "Identifier for employer group or plan sponsor.",
    "Subscriber Index": "Identifier for the policyholder.",
    "Claim Type": "Type of claim (medical, dental, pharmacy).",
    "Claim Subscriber Type": "Primary subscriber or dependent.",
    "Claim Status": "Approved, Need to Review, Likely Deniable.",
    "Network ID": "Identifier for provider network.",
    "Agreement ID": "Identifier for provider–payer contract."
}

def main(csv_path, target_col="Claim Status", outdir="artifacts"):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Features and target
    y = df[target_col]
    X = df.drop(columns=[target_col, "Claims No"])

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf = Pipeline([("pre", preprocessor), ("model", CalibratedClassifierCV(model, cv=3))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(pd.get_dummies(y_test), y_prob, average="macro")

    # Save artifacts
    joblib.dump(clf, os.path.join(outdir, "model.pkl"))
    with open(os.path.join(outdir, "training_report.json"), "w") as f:
        json.dump({"report": report, "auc": auc}, f, indent=2)
    with open(os.path.join(outdir, "feature_schema.json"), "w") as f:
        json.dump({"categorical": cat_cols, "numerical": num_cols, "descriptions": COLUMN_DESCRIPTIONS}, f, indent=2)

    print("Training complete. Artifacts saved in", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    args = parser.parse_args()
    main(args.csv_path)