import argparse
import os
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import ibm_boto3
from ibm_botocore.client import Config

# Read credentials from environment variables
api_key = os.getenv("IBM_COS_API_KEY")
service_instance_id = os.getenv("IBM_COS_SERVICE_INSTANCE_ID")

if not api_key or not service_instance_id:
    raise ValueError("Missing COS credentials. Please set IBM_COS_API_KEY and IBM_COS_SERVICE_INSTANCE_ID.")

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

def upload_to_cos(local_path, bucket, key, api_key, service_instance_id,
                  endpoint_url="https://s3.us.cloud-object-storage.appdomain.cloud"):
    cos = ibm_boto3.client("s3",
        ibm_api_key_id=api_key,
        ibm_service_instance_id=service_instance_id,
        config=Config(signature_version="oauth"),
        endpoint_url=endpoint_url
    )
    cos.upload_file(local_path, bucket, key)
    return f"cos://us-geo/{bucket}/{key}"

def run(claims_file: str, api_key: str, service_instance_id: str, outdir: str = "artifacts"):
    os.makedirs(outdir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(claims_file)
    X = df.drop(columns=["Claims No", "Status"], errors="ignore")
    y = df["Status"]

    # Train model
    clf = RandomForestClassifier()
    clf = CalibratedClassifierCV(clf)
    clf.fit(X, y)

    # Save artifacts locally
    model_path = os.path.join(outdir, "model.pkl")
    joblib.dump(clf, model_path)

    report_path = os.path.join(outdir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump({"classes": clf.classes_.tolist(),
                   "n_features": X.shape[1],
                   "n_samples": X.shape[0]}, f, indent=2)

    schema_path = os.path.join(outdir, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump({"features": list(X.columns)}, f, indent=2)

    # Upload to COS
    bucket = "output-bucket-v9mnj115br8tmdq"
    model_uri = upload_to_cos(model_path, bucket, "training/model.pkl", api_key, service_instance_id)
    report_uri = upload_to_cos(report_path, bucket, "training/training_report.json", api_key, service_instance_id)
    schema_uri = upload_to_cos(schema_path, bucket, "training/feature_schema.json", api_key, service_instance_id)

    return {
        "model_file": model_uri,
        "training_report": report_uri,
        "feature_schema": schema_uri
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims_file", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--service_instance_id", required=True)
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    outputs = run(args.claims_file, args.api_key, args.service_instance_id, args.outdir)
    print(json.dumps(outputs, indent=2))