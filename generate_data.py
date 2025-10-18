"""
generate_data.py
Generates a synthetic EHR dataset (~10,000 patients) for 30-day readmission prediction.
Saves CSV to data/simulated_ehr.csv
"""
import numpy as np
import pandas as pd
from faker import Faker
import random
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

N = 10000
os.makedirs("data", exist_ok=True)

def make_patient_record(i):
    age = int(np.clip(np.random.normal(65, 16), 18, 100))
    sex = random.choice(['Female', 'Male'])
    race = random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'])
    num_prior_adm = np.random.poisson(0.6)
    comorbidity_count = int(np.clip(np.random.poisson(2), 0, 8))
    length_of_stay = int(np.clip(np.random.exponential(4) + 1, 1, 60))
    med_count = int(np.clip(np.random.poisson(5), 0, 30))
    discharge_disposition = random.choices(
        ['Home', 'Home with Home Health', 'SNF', 'AMA', 'Other'],
        weights=[0.7, 0.15, 0.1, 0.02, 0.03]
    )[0]
    primary_dx = random.choice(['CHF', 'Pneumonia', 'UTI', 'COPD', 'Sepsis', 'Other'])
    followup_scheduled = random.choices([0, 1], weights=[0.6, 0.4])[0]

    logit = -3.0
    logit += (age - 60) * 0.02
    logit += comorbidity_count * 0.3
    logit += num_prior_adm * 0.4
    logit += (length_of_stay - 3) * 0.02
    logit += (med_count - 5) * 0.02
    if primary_dx == 'CHF':
        logit += 0.6
    if discharge_disposition == 'SNF':
        logit += 0.5
    if followup_scheduled == 1:
        logit -= 0.6
    logit += np.random.normal(0, 0.8)

    prob = 1 / (1 + np.exp(-logit))
    readmitted_30 = np.random.binomial(1, prob)

    return {
        "patient_id": f"PID{i:06d}",
        "age": age,
        "sex": sex,
        "race": race,
        "num_prior_adm": num_prior_adm,
        "comorbidity_count": comorbidity_count,
        "length_of_stay": length_of_stay,
        "med_count": med_count,
        "discharge_disposition": discharge_disposition,
        "primary_dx": primary_dx,
        "followup_scheduled": followup_scheduled,
        "readmitted_30": readmitted_30
    }

rows = [make_patient_record(i) for i in range(N)]
df = pd.DataFrame(rows)
df.to_csv("data/simulated_ehr.csv", index=False)
print("✅ Saved data/simulated_ehr.csv with", len(df), "rows")
