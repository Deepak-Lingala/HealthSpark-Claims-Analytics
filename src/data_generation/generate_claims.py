"""
HealthSpark — Synthetic Healthcare Claims Data Generator
=========================================================
Generates 500,000 realistic healthcare insurance claims and ~50,000 patient
demographic records. All data is 100% synthetic and HIPAA-safe.

Distributions are calibrated to mirror real-world healthcare claims:
  - ~12% claim denial rate (typical for commercial + Medicare mix)
  - ~15% 30-day readmission rate (aligned with CMS benchmarks)
  - Age skewed toward 45-75 (high-utilization population)
  - Top 20 ICD-10 diagnosis codes by prevalence

Usage:
    python -m src.data_generation.generate_claims
"""

import csv
import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
NUM_CLAIMS = 500_000
NUM_PATIENTS = 50_000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SEED = 42

# Top 20 ICD-10 codes by prevalence in US claims data
# Each tuple: (code, description, relative_weight, avg_los_days)
ICD10_CODES = [
    ("E11.9",  "Type 2 diabetes mellitus without complications",       12, 3),
    ("I10",    "Essential (primary) hypertension",                     15, 2),
    ("J44.1",  "COPD with acute exacerbation",                         8, 5),
    ("I50.9",  "Heart failure, unspecified",                            7, 6),
    ("J18.9",  "Pneumonia, unspecified organism",                       6, 5),
    ("M54.5",  "Low back pain",                                         9, 2),
    ("I25.10", "Atherosclerotic heart disease of native coronary artery", 5, 4),
    ("N18.9",  "Chronic kidney disease, unspecified",                   5, 4),
    ("J06.9",  "Acute upper respiratory infection, unspecified",        10, 1),
    ("E78.5",  "Hyperlipidemia, unspecified",                           8, 1),
    ("F32.9",  "Major depressive disorder, single episode, unspecified", 6, 3),
    ("K21.0",  "GERD with esophagitis",                                 7, 2),
    ("G47.33", "Obstructive sleep apnea",                               4, 1),
    ("M17.11", "Primary osteoarthritis, right knee",                    5, 3),
    ("E03.9",  "Hypothyroidism, unspecified",                           6, 1),
    ("J45.909","Unspecified asthma, uncomplicated",                     5, 2),
    ("I48.91", "Unspecified atrial fibrillation",                       4, 4),
    ("N39.0",  "Urinary tract infection, site not specified",           6, 3),
    ("K59.00", "Constipation, unspecified",                             3, 1),
    ("R10.9",  "Unspecified abdominal pain",                            4, 2),
]

# Common CPT procedure codes mapped to facility types
CPT_CODES = [
    "99213", "99214", "99215", "99223", "99232", "99233",  # E&M codes
    "99281", "99282", "99283", "99284", "99285",            # ED visits
    "43239", "27447", "33533", "47562", "49505",            # Surgical
    "71046", "74177", "93000", "93306", "70553",            # Imaging/Diagnostic
]

FACILITY_TYPES = ["Inpatient", "Outpatient", "Emergency", "Ambulatory", "SNF"]
FACILITY_WEIGHTS = [0.25, 0.35, 0.15, 0.15, 0.10]

PAYER_TYPES = ["Medicare", "Medicaid", "Commercial", "Self-Pay"]
PAYER_WEIGHTS = [0.35, 0.15, 0.40, 0.10]

INSURANCE_TYPES = ["HMO", "PPO", "EPO", "POS", "HDHP", "Medicare Advantage", "Medicaid Managed Care"]
INSURANCE_WEIGHTS = [0.20, 0.30, 0.10, 0.08, 0.12, 0.12, 0.08]

GENDERS = ["M", "F"]
GENDER_WEIGHTS = [0.48, 0.52]

US_STATES = [
    "AZ", "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC",
    "MI", "NJ", "VA", "WA", "MA", "TN", "IN", "MO", "MD", "WI",
    "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_patients(num_patients: int) -> list[dict]:
    """Generate patient demographics with realistic age distribution.

    Age is drawn from a beta distribution scaled to 18-95, peaking around 55-70
    to reflect the high-utilization population seen in real claims data.
    """
    patients = []
    # Beta distribution parameters that skew toward 55-70 age range
    ages = np.random.beta(a=5, b=3, size=num_patients) * 77 + 18  # range: 18-95

    for i in range(num_patients):
        patient_id = f"P{i+1:06d}"
        age = int(np.clip(ages[i], 18, 95))
        gender = random.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
        state = random.choice(US_STATES)
        insurance_type = random.choices(INSURANCE_TYPES, weights=INSURANCE_WEIGHTS, k=1)[0]

        # Comorbidity count correlates with age — older patients have more comorbidities
        base_comorbidity = max(0, int(np.random.poisson(lam=max(0.1, (age - 30) / 15))))
        comorbidity_count = min(base_comorbidity, 10)

        patients.append({
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "state": state,
            "insurance_type": insurance_type,
            "comorbidity_count": comorbidity_count,
        })

    return patients


def generate_claims(patients: list[dict], num_claims: int) -> list[dict]:
    """Generate healthcare claims with realistic cost, denial, and readmission distributions.

    Key design decisions:
      - Denial rate (~12%) varies by payer: Self-Pay highest, Medicare lowest
      - Readmission rate (~15%) increases with comorbidity count and certain diagnoses
      - Claim amounts follow a log-normal distribution (right-skewed, like real costs)
      - LOS varies by diagnosis and facility type
    """
    claims = []
    patient_lookup = {p["patient_id"]: p for p in patients}
    patient_ids = [p["patient_id"] for p in patients]

    # Payer-specific denial rate modifiers
    payer_denial_modifier = {
        "Medicare": 0.08,
        "Medicaid": 0.10,
        "Commercial": 0.12,
        "Self-Pay": 0.22,
    }

    # Diagnosis codes that increase readmission risk
    high_readmit_codes = {"I50.9", "J44.1", "J18.9", "N18.9", "I48.91"}

    # ICD-10 weights for random selection
    icd_codes = [c[0] for c in ICD10_CODES]
    icd_weights = [c[2] for c in ICD10_CODES]
    icd_los_map = {c[0]: c[3] for c in ICD10_CODES}

    # Date range: claims from 2022-01-01 to 2024-12-31 (3 years)
    start_date = datetime(2022, 1, 1)
    date_range_days = (datetime(2024, 12, 31) - start_date).days

    for i in range(num_claims):
        claim_id = f"CLM{i+1:07d}"
        patient_id = random.choice(patient_ids)
        patient = patient_lookup[patient_id]

        # Diagnosis and procedure
        diagnosis_code = random.choices(icd_codes, weights=icd_weights, k=1)[0]
        procedure_code = random.choice(CPT_CODES)
        provider_id = f"PRV{random.randint(1, 2000):05d}"

        # Facility type
        facility_type = random.choices(FACILITY_TYPES, weights=FACILITY_WEIGHTS, k=1)[0]

        # Payer type — correlated with insurance type
        payer_type = random.choices(PAYER_TYPES, weights=PAYER_WEIGHTS, k=1)[0]

        # Admit date and length of stay
        admit_date = start_date + timedelta(days=random.randint(0, date_range_days))
        base_los = icd_los_map[diagnosis_code]

        # LOS varies by facility: inpatient/SNF longer, outpatient/ambulatory shorter
        if facility_type in ("Inpatient", "SNF"):
            los = max(1, int(np.random.poisson(lam=base_los * 1.5)))
        elif facility_type == "Emergency":
            los = max(1, int(np.random.poisson(lam=max(1, base_los * 0.5))))
        else:
            los = max(0, int(np.random.poisson(lam=max(1, base_los * 0.3))))

        discharge_date = admit_date + timedelta(days=los)

        # Claim amount — log-normal distribution (right-skewed like real healthcare costs)
        # Higher for inpatient/surgical, lower for outpatient/ambulatory
        if facility_type == "Inpatient":
            claim_amount = round(np.random.lognormal(mean=9.0, sigma=0.8), 2)  # ~$8K median
        elif facility_type == "Emergency":
            claim_amount = round(np.random.lognormal(mean=7.5, sigma=0.9), 2)  # ~$1.8K median
        elif facility_type == "SNF":
            claim_amount = round(np.random.lognormal(mean=8.5, sigma=0.6), 2)  # ~$5K median
        else:
            claim_amount = round(np.random.lognormal(mean=6.0, sigma=1.0), 2)  # ~$400 median

        # Cap extreme outliers to keep data realistic
        claim_amount = min(claim_amount, 500_000.0)

        # Paid amount: typically 60-95% of claim amount (varies by payer)
        payment_ratio = np.random.uniform(0.60, 0.95)
        paid_amount = round(claim_amount * payment_ratio, 2)

        # Denial flag (~12% overall, varies by payer)
        denial_prob = payer_denial_modifier.get(payer_type, 0.12)
        # Higher denial for very high-cost claims (cost outlier flag)
        if claim_amount > 50_000:
            denial_prob += 0.08
        denial_flag = 1 if random.random() < denial_prob else 0

        # If denied, paid amount is $0
        if denial_flag == 1:
            paid_amount = 0.0

        # 30-day readmission (~15% overall)
        readmit_prob = 0.10
        # Higher readmission risk for certain diagnoses
        if diagnosis_code in high_readmit_codes:
            readmit_prob += 0.10
        # Higher readmission risk with more comorbidities
        readmit_prob += patient["comorbidity_count"] * 0.015
        # Cap probability
        readmit_prob = min(readmit_prob, 0.50)
        readmission_30day = 1 if random.random() < readmit_prob else 0

        claims.append({
            "claim_id": claim_id,
            "patient_id": patient_id,
            "admit_date": admit_date.strftime("%Y-%m-%d"),
            "discharge_date": discharge_date.strftime("%Y-%m-%d"),
            "diagnosis_code": diagnosis_code,
            "procedure_code": procedure_code,
            "provider_id": provider_id,
            "facility_type": facility_type,
            "payer_type": payer_type,
            "claim_amount": claim_amount,
            "paid_amount": paid_amount,
            "denial_flag": denial_flag,
            "readmission_30day": readmission_30day,
            "length_of_stay": los,
            "age": patient["age"],
            "gender": patient["gender"],
            "comorbidity_count": patient["comorbidity_count"],
            "state": patient["state"],
            "insurance_type": patient["insurance_type"],
        })

    return claims


def write_csv(data: list[dict], filepath: str) -> None:
    """Write list of dicts to CSV with headers from first record."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"  Written {len(data):,} records to {filepath}")


def main() -> None:
    print("=" * 60)
    print("HealthSpark — Synthetic Data Generation")
    print("=" * 60)

    _set_seed(SEED)

    print(f"\n[1/3] Generating {NUM_PATIENTS:,} patients...")
    patients = generate_patients(NUM_PATIENTS)

    print(f"[2/3] Generating {NUM_CLAIMS:,} claims...")
    claims = generate_claims(patients, NUM_CLAIMS)

    # Compute summary stats before writing
    denial_rate = sum(c["denial_flag"] for c in claims) / len(claims)
    readmit_rate = sum(c["readmission_30day"] for c in claims) / len(claims)
    avg_age = np.mean([c["age"] for c in claims])
    avg_claim = np.mean([c["claim_amount"] for c in claims])

    print(f"\n[3/3] Writing CSV files...")
    claims_path = os.path.join(OUTPUT_DIR, "claims.csv")
    patients_path = os.path.join(OUTPUT_DIR, "patients.csv")
    write_csv(claims, claims_path)
    write_csv(patients, patients_path)

    print(f"\n{'-' * 60}")
    print(f"Summary Statistics:")
    print(f"  Total claims:       {len(claims):,}")
    print(f"  Unique patients:    {len(patients):,}")
    print(f"  Denial rate:        {denial_rate:.1%}")
    print(f"  Readmission rate:   {readmit_rate:.1%}")
    print(f"  Average age:        {avg_age:.1f} years")
    print(f"  Average claim:      ${avg_claim:,.2f}")
    print(f"{'-' * 60}")
    print("Data generation complete.\n")


if __name__ == "__main__":
    main()
