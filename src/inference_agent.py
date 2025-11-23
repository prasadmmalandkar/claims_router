import argparse
import os
import json
import pandas as pd
import joblib
import ibm_boto3
from ibm_botocore.client import Config

# Read credentials from environment variables
api_key = os.getenv("IBM_COS_API_KEY")
service_instance_id = os.getenv("IBM_COS_SERVICE_INSTANCE_ID")

if not api_key or not service_instance_id:
    raise ValueError("Missing COS credentials. Please set IBM_COS_API_KEY and IBM_COS_SERVICE_INSTANCE_ID.")

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

def load_model_from_cos(cos_uri, api_key, service_instance_id,
                        endpoint_url="https://s3.us.cloud-object-storage.appdomain.cloud"):
    parts = cos_uri.replace("cos://", "").split("/", 2)
    bucket = parts[1]
    key = parts[2]

    cos = ibm_boto3.client("s3",
        ibm_api_key_id=api_key,
        ibm_service_instance_id=service_instance_id,
        config=Config(signature_version="oauth"),
        endpoint_url=endpoint_url
    )

    os.makedirs("artifacts", exist_ok=True)
    local_path = os.path.join("artifacts", os.path.basename(key))
    cos.download_file(bucket, key, local_path)
    return local_path

def run(new_encounters: str, model_uri: str, api_key: str, service_instance_id: str,
        schema_file: str = None, outdir: str = "artifacts"):

    os.makedirs(outdir, exist_ok=True)

    # Load model from COS
    model_path = load_model_from_cos(model_uri, api_key, service_instance_id)
    model = joblib.load(model_path)

    # Load new encounters
    df = pd.read_csv(new_encounters)
    X = df.drop(columns=["Claims No"], errors="ignore")

    preds = model.predict(X)
    df["Predicted Status"] = preds

    denial_map = {
        "Approved": "No denial reason – claim approved",
        "Need to Review": "Requires manual review – missing documentation or prior authorization",
        "Likely Deniable": "High risk denial – non‑covered diagnosis or coding mismatch"
    }
    df["Denial Reason Code"] = df["Predicted Status"].map(denial_map)

    # Save outputs locally
    scored_path = os.path.join(outdir, "scored_encounters.csv")
    df.to_csv(scored_path, index=False)

    approved_path = os.path.join(outdir, "approved.txt")
    need_review_path = os.path.join(outdir, "need_review.txt")
    likely_deniable_path = os.path.join(outdir, "likely_deniable.txt")

    df[df["Predicted Status"] == "Approved"].to_csv(approved_path, index=False)
    df[df["Predicted Status"] == "Need to Review"].to_csv(need_review_path, index=False)
    df[df["Predicted Status"] == "Likely Deniable"].to_csv(likely_deniable_path, index=False)

    report_path = os.path.join(outdir, "inference_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "prediction_counts": df["Predicted Status"].value_counts().to_dict(),
            "denial_reason_map": denial_map
        }, f, indent=2)

    # Upload to COS
    bucket = "output-bucket-v9mnj115br8tmdq"
    scored_uri = upload_to_cos(scored_path, bucket, "inference/scored_encounters.csv", api_key, service_instance_id)
    approved_uri = upload_to_cos(approved_path, bucket, "inference/approved.txt", api_key, service_instance_id)
    need_review_uri = upload_to_cos(need_review_path, bucket, "inference/need_review.txt", api_key, service_instance_id)
    likely_deniable_uri = upload_to_cos(likely_deniable_path, bucket, "inference/likely_deniable.txt", api_key, service_instance_id)
    report_uri = upload_to_cos(report_path, bucket, "inference/inference_report.json", api_key, service_instance_id)

    return {
        "scored_encounters": scored_uri,
        "approved_file": approved_uri,
        "need_review_file": need_review_uri,
        "likely_deniable_file": likely_deniable_uri,
        "inference_report": report_uri
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_encounters", required=True)
    parser.add_argument("--model_uri", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--service_instance_id", required=True)
    parser.add_argument("--schema_file", default=None)
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    outputs = run(args.new_encounters, args.model_uri, args.api_key, args.service_instance_id,
                  args.schema_file, args.outdir)
    print(json.dumps(outputs, indent=2))