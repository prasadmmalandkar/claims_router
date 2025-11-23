import pandas as pd
import random, os

num_encounters = 100

line_of_business_ids = ["LOB001", "LOB002", "LOB003"]
revenue_codes = ["0450", "0300", "0250", "0510"]
service_codes = ["SVC01", "SVC02", "SVC03"]
procedure_codes = ["99213", "97110", "93000", "70553"]
procedure_descriptions = {
    "99213": "Office/outpatient visit",
    "97110": "Therapeutic exercises",
    "93000": "Electrocardiogram",
    "70553": "MRI brain with contrast"
}
diagnosis_codes = ["I10", "M25.50", "R07.9", "E11.9"]
claim_types = ["Medical", "Dental", "Pharmacy"]
subscriber_types = ["Primary", "Dependent"]

data = []
for i in range(1, num_encounters + 1):
    proc_code = random.choice(procedure_codes)
    row = {
        "Claims No": f"NEW{i:04d}",
        "Claim Line": random.randint(1, 5),
        "Member ID": f"M{random.randint(1000,9999)}",
        "Provider ID": f"P{random.randint(1000,9999)}",
        "Line of Business ID": random.choice(line_of_business_ids),
        "Revenue Code": random.choice(revenue_codes),
        "Service Code": random.choice(service_codes),
        "Procedure Code": proc_code,
        "Procedure Description": procedure_descriptions[proc_code],
        "Diagnosis": random.choice(diagnosis_codes),
        "Claim Charge Amount": round(random.uniform(100, 5000), 2),
        "Denial Reason Code": "PENDING",   # placeholder
        "Price Index": random.randint(1, 10),
        "In / Out Network": random.choice(["In", "Out"]),
        "Reference Index": random.randint(100, 999),
        "Subscriber Payment Amount": round(random.uniform(20, 500), 2),
        "Insurance Eligible Amount": round(random.uniform(50, 4000), 2),
        "Group Index": random.randint(1, 20),
        "Subscriber Index": random.randint(1, 50),
        "Claim Type": random.choice(claim_types),
        "Claim Subscriber Type": random.choice(subscriber_types),
        "Network ID": f"N{random.randint(100,999)}",
        "Agreement ID": f"A{random.randint(1000,9999)}"
    }
    data.append(row)

df = pd.DataFrame(data)

os.makedirs("data/raw", exist_ok=True)
output_path = "data/raw/new_encounters.csv"
df.to_csv(output_path, index=False)

print(f"✅ Generated {num_encounters} new patient encounters at {output_path}")