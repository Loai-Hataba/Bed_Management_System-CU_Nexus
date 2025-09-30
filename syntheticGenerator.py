#!/usr/bin/env python3
"""
generate_synthetic_neuro.py

Rationale / header:
This script creates a reproducible synthetic neurology department SQLite database suitable for ML prototyping.
It intentionally injects noise and missingness to mimic real clinical datasets and avoid overfitting.

Limitations: This data is entirely synthetic and NOT PHI. Use only for prototyping and modeling — not clinical decision making.

Primary features implemented (per the exhaustive specification provided):
- Creates all tables exactly as required (with PRAGMA foreign_keys = ON).
- Generates NUM_PATIENTS patients and ~1200 admissions across 2 years.
- Inserts realistic labs, investigations, imaging results, neurological scales, medications, comorbidities, risk factors, preop checklist, and patientfeatures rows for discharged admissions only.
- Applies 15% missingness for certain labs, 5% measurement noise, and 2% long-stay outliers.
- Uses a fixed random seed (0) for reproducibility.
- Validations printed at the end.

If any numeric constant was adjusted for robustness, it is documented inline where that occurs. No external network access is required. Requires Python 3.10+ and numpy.

Run as:
    python generate_synthetic_neuro.py [--db-path PATH] [--append] [--export-csv]

Example:
    python generate_synthetic_neuro.py --db-path /mnt/data/synthetic_neuro.db --export-csv

"""

import os
import sys
import sqlite3
import random
import argparse
import csv
import traceback
from datetime import datetime, timedelta
import numpy as np


# ----------------------
# Top-level defaults (exact values from spec)
# ----------------------
NUM_PATIENTS = 1000
YEARS = 2
NUM_BEDS = 100
NUM_ROOMS = 5
NUM_MEDICATIONS = 30
MISSING_RATE_LAB = 0.15
SEED = 0
DB_DEFAULT_PATH = 'RFID.db'
EXPORT_CSV_DEFAULT = False

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Diagnosis distribution (exact frequencies)
DIAGNOSES = [
    ("Ischemic stroke", 0.30),
    ("Hemorrhagic stroke", 0.10),
    ("Brain tumor", 0.15),
    ("Epilepsy/Seizure", 0.15),
    ("Migraine/Headache", 0.10),
    ("Parkinsons/Movement disorder", 0.05),
    ("Other", 0.15),
]

DIAGNOSIS_LIST = [d for d, p in DIAGNOSES]
DIAGNOSIS_WEIGHTS = [p for d, p in DIAGNOSES]
DIAGNOSIS_MEDIANS = {
    'Ischemic stroke': 10,
    'Hemorrhagic stroke': 15,
    'Brain tumor': 12,
    'Epilepsy/Seizure': 4,
    'Migraine/Headache': 2,
    'Parkinsons/Movement disorder': 8,
    'Other': 7,
}

# Diagnosis-specific base severity ranges (exact)
BASE_SEVERITY_RANGES = {
    'Ischemic stroke': (6,12),
    'Hemorrhagic stroke': (8,20),
    'Brain tumor': (5,15),
    'Epilepsy/Seizure': (2,8),
    'Migraine/Headache': (1,3),
    'Parkinsons/Movement disorder': (3,10),
    'Other': (2,12),
}

# Lab ranges (exact values)
LAB_RANGES = {
    'Sodium': {'low':135, 'high':145, 'unit':'mEq/L', 'code':'SODIUM'},
    'Potassium': {'low':3.5, 'high':5.1, 'unit':'mEq/L', 'code':'POTASSIUM'},
    'Creatinine': {'low':0.6, 'high':1.3, 'unit':'mg/dL', 'code':'CREATININE'},
    'GFR': {'low':60, 'high':120, 'unit':'mL/min/1.73m2', 'code':'GFR'},
    'ALT': {'low':7, 'high':56, 'unit':'U/L', 'code':'ALT'},
    'AST': {'low':10, 'high':40, 'unit':'U/L', 'code':'AST'},
    'Bilirubin': {'low':0.0, 'high':1.2, 'unit':'mg/dL', 'code':'BILIRUBIN'},
    'Hemoglobin': {'low':12.0, 'high':17.5, 'unit':'g/dL', 'code':'HEMOGLOBIN'},
    'WBC': {'low':4.0, 'high':11.0, 'unit':'10^3/uL', 'code':'WBC'},
    'Platelets': {'low':150, 'high':450, 'unit':'10^3/uL', 'code':'PLATELETS'},
    'Blood_Sugar': {'low':70, 'high':140, 'unit':'mg/dL', 'code':'BLOOD_SUGAR'},
    'PT': {'low':11.0, 'high':13.5, 'unit':'sec', 'code':'PT'},
    'INR': {'low':0.8, 'high':1.2, 'unit':'ratio', 'code':'INR'},
    'APTT': {'low':25.0, 'high':40.0, 'unit':'sec', 'code':'APTT'},
}

# Medication list (30 realistic names spanning requested classes)
MEDICATION_NAMES = [
    'Aspirin', 'Clopidogrel', 'Warfarin', 'Apixaban', 'Enoxaparin',
    'Levetiracetam', 'Valproate', 'Carbamazepine', 'Phenytoin', 'Lamotrigine',
    'Atorvastatin', 'Simvastatin', 'Rosuvastatin', 'Metformin', 'Insulin',
    'Lisinopril', 'Amlodipine', 'Losartan', 'Metoprolol', 'Propranolol',
    'Prednisone', 'Dexamethasone', 'Ibuprofen', 'Paracetamol', 'Tramadol',
    'Gabapentin', 'Pregabalin', 'Diazepam', 'Nitrofurantoin', 'Omeprazole'
]
# Ensure length exactly NUM_MEDICATIONS
if len(MEDICATION_NAMES) != NUM_MEDICATIONS:
    # If the list length differs, adjust by trimming or repeating with suffixes (documented change)
    if len(MEDICATION_NAMES) < NUM_MEDICATIONS:
        # Append numbered duplicates (explicitly documented change for robustness)
        for i in range(NUM_MEDICATIONS - len(MEDICATION_NAMES)):
            MEDICATION_NAMES.append(f"Med_{i+1}")
    else:
        MEDICATION_NAMES = MEDICATION_NAMES[:NUM_MEDICATIONS]

# InvestigationTypes codes to populate exactly (list from spec)
INVESTIGATION_TYPES = [
    ('SODIUM','Sodium','lab'), ('POTASSIUM','Potassium','lab'), ('CREATININE','Creatinine','lab'),
    ('GFR','GFR','lab'), ('ALT','ALT','lab'), ('AST','AST','lab'), ('BILIRUBIN','Bilirubin','lab'),
    ('HEMOGLOBIN','Hemoglobin','lab'), ('WBC','WBC','lab'), ('PLATELETS','Platelets','lab'),
    ('BLOOD_SUGAR','Blood Sugar','lab'), ('PT','PT','lab'), ('INR','INR','lab'), ('APTT','APTT','lab'),
    ('ECG','ECG','procedure'), ('CT_BRAIN','CT Brain','imaging'), ('MRI_BRAIN','MRI Brain','imaging'),
    ('ANGIO','Angiography','imaging'), ('EEG','EEG','neuro'), ('GCS','GCS','scale'), ('WFNS','WFNS','scale'), ('STOPBANG','STOPBANG','scale')
]

IMAGING_TYPES = [
    ('CT_BRAIN','CT Brain'), ('MRI_BRAIN','MRI Brain'), ('ANGIO','Angiography'), ('DOPPLER','Doppler')
]

SURGEONS = ['Dr. Ahmed El-Sayed','Dr. Mohamed Hassan','Dr. Sara Ali','Dr. Karim Mostafa','Dr. Yasmine Farid']

# Egyptian-style name components (small curated lists to produce plausible Egyptian names)
MALE_FIRST = ['Ahmed','Mohamed','Mostafa','Omar','Ali','Amr','Hassan','Youssef','Khaled','Mahmoud','Tarek','Walid','Ayman','Hisham','Ibrahim','Samir','Fady','Hussein','Nader','Saeed']
FEMALE_FIRST = ['Sara','Fatma','Mariam','Yasmine','Nour','Aisha','Hala','Rana','Dina','Mona','Laila','Reem','Mai','Asmaa','Rita','Salma','Noha','Nadia','Hana','Lina']
LAST_NAMES = ['El-Sayed','Hassan','Mohamed','Ali','Ibrahim','Farid','Mahmoud','Nabil','Fahmy','Said','Mostafa','Kamel','Abdelrahman','Gamal','Zaki','Youssef','Khalil','Hussein','Tawfik','Saber']

# ----------------------
# Utility functions
# ----------------------

def iso_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def random_phone():
    """Generate an Egyptian-style mobile number (11 digits starting with allowed prefixes)."""
    prefix = random.choice(['010','011','012','015'])
    rest = ''.join(str(random.randint(0,9)) for _ in range(8))
    return prefix + rest


def clip(val, low, high):
    return max(low, min(high, val))


def make_uid(index : str) -> str:
    # short unique id — deterministic given seed because uuid4 uses randomness; use uuid4 but also random seed set
    return f"P{index:04d}"

# ----------------------
# Schema creation
# ----------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS Patients(
    UID TEXT PRIMARY KEY UNIQUE,
    Name TEXT,
    Phone TEXT,
    Age INTEGER NOT NULL,
    Sex TEXT,
    BloodType TEXT NOT NULL,
    Admission_reason TEXT,
    Family_history_disease TEXT,
    Diagnosis TEXT,
    Expected_Recovery TEXT
);

CREATE TABLE IF NOT EXISTS PatientMedication(
    patient_id TEXT REFERENCES Patients(UID),
    medicine_id INTEGER REFERENCES Medications(medicine_id),
    current_medication INTEGER,
    PRIMARY KEY (patient_id, medicine_id)
);

CREATE TABLE IF NOT EXISTS Medications(
    medicine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine_name TEXT,
    available INTEGER
);

CREATE TABLE IF NOT EXISTS Beds(
    Bed_id INTEGER PRIMARY KEY AUTOINCREMENT,
    Room_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS Admissions (
  admission_id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id TEXT NOT NULL REFERENCES Patients(UID),
  bed_id INTEGER REFERENCES Beds(Bed_id),
  admission_date TEXT NOT NULL,
  discharge_date TEXT,
  admission_reason TEXT,
  diagnosis TEXT,
  status TEXT CHECK(status IN ('admitted','discharged','transferred','expired')) DEFAULT 'admitted',
  primary_surgeon TEXT,
  preop_done INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS DiagnosisCodes (
  code TEXT PRIMARY KEY,
  description TEXT
);

CREATE TABLE IF NOT EXISTS InvestigationTypes (
  type_id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT UNIQUE,
  name TEXT,
  category TEXT
);

CREATE TABLE IF NOT EXISTS ImagingTypes (
  imaging_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT UNIQUE,
  name TEXT
);

CREATE TABLE IF NOT EXISTS Investigations(
  investigation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER REFERENCES Admissions(admission_id),
  patient_id TEXT NOT NULL REFERENCES Patients(UID),
  type_id INTEGER REFERENCES InvestigationTypes(type_id),
  result_text TEXT,
  result_value REAL,
  unit TEXT,
  abnormal INTEGER DEFAULT 0,
  date TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS LabResults (
  lab_id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER REFERENCES Admissions(admission_id),
  patient_id TEXT NOT NULL REFERENCES Patients(UID),
  test_name TEXT,
  test_code TEXT,
  value REAL,
  unit TEXT,
  normal_low REAL,
  normal_high REAL,
  abnormal INTEGER DEFAULT 0,
  date TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS NeurologicalScales (
  scale_id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER REFERENCES Admissions(admission_id),
  patient_id TEXT NOT NULL REFERENCES Patients(UID),
  gcs_eye INTEGER,
  gcs_verbal INTEGER,
  gcs_motor INTEGER,
  gcs_total INTEGER,
  wfns_grade INTEGER,
  stop_snore INTEGER,
  stop_tired INTEGER,
  stop_observed INTEGER,
  stop_bp INTEGER,
  stop_bmi INTEGER,
  stop_age INTEGER,
  stop_neck INTEGER,
  stop_gender INTEGER,
  stopbang_score INTEGER,
  date TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS Comorbidities (
  patient_id TEXT REFERENCES Patients(UID),
  admission_id INTEGER REFERENCES Admissions(admission_id),
  comorbidity TEXT,
  present INTEGER DEFAULT 0,
  PRIMARY KEY (patient_id, admission_id, comorbidity)
);

CREATE TABLE IF NOT EXISTS RiskFactors (
  factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER REFERENCES Admissions(admission_id),
  patient_id TEXT NOT NULL REFERENCES Patients(UID),
  bmi REAL,
  neck_circumference REAL,
  smoker INTEGER DEFAULT 0,
  pregnancy INTEGER DEFAULT 0,
  allergy_contrast INTEGER DEFAULT 0,
  airway_difficulty INTEGER DEFAULT 0,
  date TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS PreoperativeChecklist (
  checklist_id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER REFERENCES Admissions(admission_id),
  patient_id TEXT NOT NULL REFERENCES Patients(UID),
  bp_systolic INTEGER,
  bp_diastolic INTEGER,
  ecg_abnormal INTEGER DEFAULT 0,
  airway_assessment TEXT,
  pregnancy_test INTEGER DEFAULT 0,
  coag_pt REAL,
  coag_inr REAL,
  coag_aptt REAL,
  blood_sugar REAL,
  date TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS PatientFeatures (
  feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER REFERENCES Admissions(admission_id),
  patient_id TEXT REFERENCES Patients(UID),
  age INTEGER,
  sex TEXT,
  bmi REAL,
  diagnosis TEXT,
  blood_type TEXT,
  gcs_total INTEGER,
  wfns_grade INTEGER,
  stopbang_score INTEGER,
  sodium REAL,
  potassium REAL,
  creatinine REAL,
  gfr REAL,
  alt REAL,
  ast REAL,
  bilirubin REAL,
  hemoglobin REAL,
  wbc REAL,
  platelets REAL,
  blood_sugar REAL,
  num_medications INTEGER,
  num_investigations INTEGER,
  imaging_abnormal_count INTEGER,
  comorbidity_count INTEGER,
  severity_score INTEGER,
  admission_date TEXT,
  discharge_date TEXT,
  recovery_days INTEGER,
  predicted_recovery_days INTEGER,
  predicted_discharge_date TEXT,
  model_version TEXT,
  prediction_confidence REAL,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS Predictions (
  prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER REFERENCES Admissions(admission_id),
  patient_id TEXT REFERENCES Patients(UID),
  predicted_recovery_days INTEGER,
  predicted_discharge_date TEXT,
  model_version TEXT,
  confidence REAL,
  created_at TEXT DEFAULT (datetime('now'))
);
"""

# ----------------------
# Main generation functions
# ----------------------

def create_schema(conn: sqlite3.Connection):
    """Create all tables and indexes. Uses PRAGMA foreign_keys = ON.

    This function creates the schema exactly as provided in the spec and also
    creates a few helpful indexes for performance.
    """
    cur = conn.cursor()
    print("Setting PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA foreign_keys = ON;")
    print("Creating tables...")
    cur.executescript(SCHEMA_SQL)
    # Indexes to speed up lookups
    print("Creating indexes...")
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_adm_patient ON Admissions(patient_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_lab_adm_test ON LabResults(admission_id, test_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_inv_adm ON Investigations(admission_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scale_adm ON NeurologicalScales(admission_id);")
    except sqlite3.Error as e:
        print("Warning: could not create one or more indexes:", e)
    conn.commit()


def populate_medications(conn: sqlite3.Connection):
    """Populate the Medications table with NUM_MEDICATIONS names and availability flags."""
    print(f"Populating Medications ({NUM_MEDICATIONS})...")
    cur = conn.cursor()
    for med in MEDICATION_NAMES:
        # Make most medications available (1) but randomly mark a few as unavailable to simulate stockouts
        available = 1 if random.random() > 0.05 else 0
        cur.execute("INSERT OR IGNORE INTO Medications (medicine_name, available) VALUES (?,?);", (med, available))
    conn.commit()


def populate_investigation_and_imaging_types(conn: sqlite3.Connection):
    """Populate InvestigationTypes and ImagingTypes from the lists defined above.
    Uses INSERT OR IGNORE so repeated runs are idempotent.
    """
    print("Populating InvestigationTypes and ImagingTypes...")
    cur = conn.cursor()
    for code, name, category in INVESTIGATION_TYPES:
        cur.execute("INSERT OR IGNORE INTO InvestigationTypes (code, name, category) VALUES (?,?,?);", (code, name, category))
    for code, name in IMAGING_TYPES:
        cur.execute("INSERT OR IGNORE INTO ImagingTypes (code, name) VALUES (?,?);", (code, name))
    conn.commit()


def populate_beds(conn: sqlite3.Connection, num_beds=NUM_BEDS, num_rooms=NUM_ROOMS):
    """Create NUM_BEDS rows and distribute them across num_rooms cyclically by Room_id.

    Note: For simplicity we do not enforce simultaneous occupancy constraints later when assigning beds.
    """
    print(f"Creating {num_beds} Beds across {num_rooms} rooms...")
    cur = conn.cursor()
    for i in range(num_beds):
        room_id = (i % num_rooms) + 1
        cur.execute("INSERT OR IGNORE INTO Beds (Bed_id, Room_id) VALUES (?,?);", (i+1, room_id))
    conn.commit()


def generate_patients(conn: sqlite3.Connection, n=NUM_PATIENTS):
    """Generate n patients with Egyptian-style names, phones, ages, sex, blood type and diagnosis.

    - Age is sampled as round(clip(Normal(62,18),0,95)).
    - Sex roughly 50/50.
    - Pregnancy and neck, BMI handled later per-admission in RiskFactors.
    """
    print(f"Creating Patients ({n}) ...")
    cur = conn.cursor()
    blood_types = ["A+","A-","B+","B-","O+","O-","AB+","AB-"]
    patients = []
    for i in range(n):
        uid = make_uid(i)
        sex = random.choice(['M','F'])
        if sex == 'M':
            fname = random.choice(MALE_FIRST)
        else:
            fname = random.choice(FEMALE_FIRST)
        lname = random.choice(LAST_NAMES)
        name = f"{fname} {lname}"
        phone = random_phone()
        # Age distribution skewed older
        age = int(round(clip(np.random.normal(62,18), 0, 95)))
        blood = random.choice(blood_types)
        diagnosis = random.choices(DIAGNOSIS_LIST, weights=DIAGNOSIS_WEIGHTS, k=1)[0]
        cur.execute(
            "INSERT OR IGNORE INTO Patients (UID, Name, Phone, Age, Sex, BloodType, Diagnosis) VALUES (?,?,?,?,?,?,?);",
            (uid, name, phone, age, sex, blood, diagnosis)
        )
        patients.append({'uid':uid, 'age':age, 'sex':sex, 'bmi':None, 'name':name, 'diagnosis':diagnosis})
    conn.commit()
    return patients


def _sample_admission_date(start_date: datetime, end_date: datetime):
    """Sample uniformly between start_date and end_date (inclusive)."""
    total_seconds = int((end_date - start_date).total_seconds())
    # If range is zero (shouldn't be), return start
    if total_seconds <= 0:
        return start_date
    delta = random.randint(0, total_seconds)
    return start_date + timedelta(seconds=delta)


def generate_admissions_and_related(conn: sqlite3.Connection, patients, years=YEARS):
    """For each patient, generate 1 admission and 20% chance of a second admission. Populate related tables.

    Returns a list of admission metadata dicts created for further processing.
    """
    print("Generating Admissions, Comorbidities, RiskFactors, NeurologicalScales, LabResults, Investigations, Preop, and PatientMedication...")
    cur = conn.cursor()

    now = datetime.now()
    start = now - timedelta(days=int(365*years))

    admissions_created = []

    # Get medication IDs to choose from later
    cur.execute("SELECT medicine_id FROM Medications;")
    meds_all = [r[0] for r in cur.fetchall()]
    if not meds_all:
        raise RuntimeError("No medications found. Populate medications first.")


    # We'll build a list of admission insert statements to batch inside a transaction
    admissions_to_insert = []

    for p in patients:
        # decide number of admissions (1 or 2) with 20% chance of second
        n_adm = 1 + (1 if random.random() < 0.20 else 0)
        first_adm_date = None
        for adm_idx in range(n_adm):
            # Sample admission_date uniformly in 2-year window
            if adm_idx == 0:
                adm_dt = _sample_admission_date(start, now)
                first_adm_date = adm_dt
            else:
                # for second admission, sample a date after first admission (add at least 1 day), but still within window
                min_dt = first_adm_date + timedelta(days=1)
                if min_dt > now:
                    # improbable, but if first_adm_date is too close to now, fallback to sample between start and now
                    adm_dt = _sample_admission_date(start, now)
                else:
                    adm_dt = _sample_admission_date(min_dt, now)
            admission_date_str = adm_dt.strftime('%Y-%m-%d %H:%M:%S')

            # Admission reason default to patient's diagnosis
            admission_reason = p.get('diagnosis')
            diagnosis = p.get('diagnosis')

            # Select bed (we allow reuse historically; not enforcing simultaneous occupancy)
            bed_id = random.randint(1, NUM_BEDS)

            # Primary surgeon
            surgeon = random.choice(SURGEONS)

            # preop_done 85% chance
            preop_done = 1 if random.random() < 0.85 else 0

            # For now set discharge_date and status to be filled after computing recovery_days
            admissions_to_insert.append((p['uid'], bed_id, admission_date_str, None, admission_reason, diagnosis, 'admitted', surgeon, preop_done))

    # Bulk insert admissions and get their generated admission_ids
    print(f"Inserting {len(admissions_to_insert)} admissions (expected ~{int(len(patients)*1.2)}) ...")
    cur.execute("BEGIN TRANSACTION;")
    for adm in admissions_to_insert:
        cur.execute(
            "INSERT INTO Admissions (patient_id, bed_id, admission_date, discharge_date, admission_reason, diagnosis, status, primary_surgeon, preop_done) VALUES (?,?,?,?,?,?,?,?,?);",
            adm
        )
    conn.commit()

    # Fetch all admissions we just created (we assume no other admissions in DB for simplicity)
    cur.execute("SELECT admission_id, patient_id, admission_date, bed_id, diagnosis, preop_done FROM Admissions;")
    all_adms = cur.fetchall()

    # Now iterate admissions to generate all dependent records (comorbidities, risk factors, labs, scales, imaging, preop, medications)
    lab_rows = 0
    inv_rows = 0
    scales_rows = 0
    comorb_rows = 0
    risk_rows = 0
    preop_rows = 0
    med_assignments = 0

    # Precompute comorbidity prevalences
    comorb_list = [
        ('hypertension', 0.40),
        ('diabetes', 0.20),
        ('ischemic_heart_disease', 0.10),
        ('chronic_kidney_disease', 0.05),
        ('copd', 0.08),
    ]

    # Put everything inside a single large transaction for speed
    cur.execute("BEGIN TRANSACTION;")

    for adm_row in all_adms:
        admission_id, patient_id, admission_date_str, bed_id, diagnosis, preop_done = adm_row[0], adm_row[1], adm_row[2], adm_row[3], adm_row[4], adm_row[5]
        # Convert admission_date_str to datetime for arithmetic
        admission_dt = datetime.strptime(admission_date_str, '%Y-%m-%d %H:%M:%S')

        # lookup patient age and sex
        cur.execute("SELECT Age, Sex FROM Patients WHERE UID = ?;", (patient_id,))
        r = cur.fetchone()
        age = r[0]
        sex = r[1]

        # BMI and neck: BMI sampled per admission (as patients may have different measurements per admission)
        bmi = float(clip(np.random.normal(27,5), 12, 50))
        # neck ≈ 35 + 0.5*(BMI - 25) + N(0,2), clipped to 25–60 cm
        neck = float(clip(35 + 0.5*(bmi - 25) + np.random.normal(0,2), 25, 60))

        # Comorbidities
        comorbidity_count = 0
        for com_name, preval in comorb_list:
            present = 1 if random.random() < preval else 0
            cur.execute("INSERT OR IGNORE INTO Comorbidities (patient_id, admission_id, comorbidity, present) VALUES (?,?,?,?);", (patient_id, admission_id, com_name, present))
            comorb_rows += 1
            if present:
                comorbidity_count += 1

        # RiskFactors row
        smoker = 1 if random.random() < 0.20 else 0
        pregnancy = 0
        if sex == 'F' and 12 <= age <= 50 and random.random() < 0.08:
            pregnancy = 1
        allergy_contrast = 1 if random.random() < 0.03 else 0
        airway_difficulty = 1 if random.random() < 0.05 else 0
        cur.execute(
            "INSERT INTO RiskFactors (admission_id, patient_id, bmi, neck_circumference, smoker, pregnancy, allergy_contrast, airway_difficulty, date) VALUES (?,?,?,?,?,?,?,?,?);",
            (admission_id, patient_id, bmi, neck, smoker, pregnancy, allergy_contrast, airway_difficulty, admission_date_str)
        )
        risk_rows += 1

        # PreoperativeChecklist
        bp_systolic = int(round(clip(np.random.normal(130,20), 80, 220)))
        bp_diastolic = int(round(clip(np.random.normal(80,12), 40, 140)))
        ecg_abnormal = 1 if random.random() < 0.08 else 0
        airway_assessment = random.choice(['easy','difficult','unknown'])
        pregnancy_test = pregnancy if sex == 'F' else 0
        coag_pt = float(clip(np.random.normal(12.0, 0.5), 8.0, 30.0))
        coag_inr = float(clip(np.random.normal(1.0, 0.05), 0.5, 5.0))
        coag_aptt = float(clip(np.random.normal(32.5, 3.0), 10.0, 120.0))
        blood_sugar = float(clip(np.random.normal(100,30), 40, 400))
        cur.execute(
            "INSERT INTO PreoperativeChecklist (admission_id, patient_id, bp_systolic, bp_diastolic, ecg_abnormal, airway_assessment, pregnancy_test, coag_pt, coag_inr, coag_aptt, blood_sugar, date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",
            (admission_id, patient_id, bp_systolic, bp_diastolic, ecg_abnormal, airway_assessment, pregnancy_test, coag_pt, coag_inr, coag_aptt, blood_sugar, admission_date_str)
        )
        preop_rows += 1

        # Severity computation preliminary: base severity drawn from diagnosis-specific range
        base_min, base_max = BASE_SEVERITY_RANGES.get(diagnosis, (2,12))
        base_severity = random.randint(base_min, base_max)
        age_factor = int((age - 60) / 20)
        severity_score = int(clip(round(base_severity + age_factor + np.random.normal(0,1)), 1, 20))

        # For discharged admissions compute recovery_days; we'll decide discharged/admitted later.
        diagnosis_median = DIAGNOSIS_MEDIANS.get(diagnosis, 7)
        recovery_days_sample = int(max(1, round(np.random.normal(loc=severity_score*1.5, scale=3) + diagnosis_median)))
        # 2% chance of long-stay outlier (+10-60 days)
        if random.random() < 0.02:
            extra = random.randint(10,60)
            recovery_days_sample += extra

        # Discharge date if discharged will be admission + recovery_days_sample days (rounded)
        tentative_discharge_dt = admission_dt + timedelta(days=int(recovery_days_sample))

        # Decide intended status to meet ~80% discharged (sample then fix future date issues)
        intended_discharged = True if random.random() < 0.80 else False
        status = 'admitted'
        discharge_date_str = None
        actual_recovery_days = None

        if intended_discharged:
            # Only mark discharged if the discharge date would not be in the future
            if tentative_discharge_dt <= datetime.now():
                status = 'discharged'
                discharge_date_str = tentative_discharge_dt.strftime('%Y-%m-%d %H:%M:%S')
                actual_recovery_days = (tentative_discharge_dt - admission_dt).days
                if actual_recovery_days < 1:
                    actual_recovery_days = 1
            else:
                # Edge case: computed discharge date in the future -> reclassify as admitted (per spec)
                status = 'admitted'
                discharge_date_str = None
                actual_recovery_days = None
        else:
            status = 'admitted'
            discharge_date_str = None
            actual_recovery_days = None

        # Update the admission row with status/discharge_date
        cur.execute("UPDATE Admissions SET status = ?, discharge_date = ? WHERE admission_id = ?;", (status, discharge_date_str, admission_id))

        # NeurologicalScales: compute GCS components derived from severity_score
        base_gcs = max(3, 15 - int(severity_score / 1.5))
        # Sample components in small window around base_gcs but within valid ranges
        # eye 1-4, verbal 1-5, motor 1-6
        # We'll distribute base_gcs across components roughly proportional to their maxes
        # Start with total GCS target = base_gcs, then sample components that sum around that target
        # Simple heuristic: sample each component around its max scaled by base_gcs/15
        eye = int(clip(round(np.random.normal(loc=clip(base_gcs*(4/15),1,4), scale=0.6)), 1, 4))
        verbal = int(clip(round(np.random.normal(loc=clip(base_gcs*(5/15),1,5), scale=0.8)), 1, 5))
        motor = int(clip(round(np.random.normal(loc=clip(base_gcs*(6/15),1,6), scale=1.0)), 1, 6))
        gcs_total = eye + verbal + motor

        wfns_grade = None
        if 'Hemorrhagic' in diagnosis:
            # WFNS meaningful for hemorrhagic stroke, sample 1-5 with mean ~3
            wfns_grade = int(clip(round(np.random.normal(3,1)),1,5))
        else:
            wfns_grade = None

        # STOPBANG components
        stop_snore = 1 if random.random() < 0.2 else 0
        stop_tired = 1 if random.random() < 0.2 else 0
        stop_observed = 1 if random.random() < 0.1 else 0
        stop_bp = 1 if bp_systolic > 140 or bp_diastolic > 90 else 0
        stop_bmi = 1 if bmi > 35 else 0
        stop_age = 1 if age > 50 else 0
        stop_neck = 1 if neck > 40 else 0
        stop_gender = 1 if sex == 'M' else 0
        stopbang_score = sum([stop_snore, stop_tired, stop_observed, stop_bp, stop_bmi, stop_age, stop_neck, stop_gender])

        # Insert NeurologicalScales row
        cur.execute(
            "INSERT INTO NeurologicalScales (admission_id, patient_id, gcs_eye, gcs_verbal, gcs_motor, gcs_total, wfns_grade, stop_snore, stop_tired, stop_observed, stop_bp, stop_bmi, stop_age, stop_neck, stop_gender, stopbang_score, date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);",
            (admission_id, patient_id, eye, verbal, motor, gcs_total, wfns_grade, stop_snore, stop_tired, stop_observed, stop_bp, stop_bmi, stop_age, stop_neck, stop_gender, stopbang_score, admission_date_str)
        )
        scales_rows += 1

        # LabResults: for each lab in LAB_RANGES, with MISSING_RATE skip; apply noise and small bias for high severity
        abnormal_lab_count = 0
        for lab_name, info in LAB_RANGES.items():
            if random.random() < MISSING_RATE_LAB:
                # simulate missing lab -> skip
                continue
            low = info['low']
            high = info['high']
            value = float(np.random.uniform(low, high))
            # measurement noise: multiply by (1 + Normal(0, 0.05))
            noise_factor = 1.0 + float(np.random.normal(0, 0.05))
            value = value * noise_factor
            # Severity bias: if severity_score > 12 bias some labs toward abnormal
            if severity_score > 12:
                if lab_name == 'Creatinine':
                    value *= 1.2
                if lab_name == 'GFR':
                    value *= 0.9  # GFR down
                if lab_name == 'WBC':
                    value *= 1.15
            # Clip to plausible physical bounds to avoid impossible values
            # For some labs negative values are impossible
            if lab_name in ['Sodium','Potassium','Creatinine','GFR','ALT','AST','Bilirubin','Hemoglobin','WBC','Platelets','Blood_Sugar','PT','INR','APTT']:
                # We'll allow some small leeway beyond specified but clip hard to reasonable min/max
                min_phys = 0.0001
                max_phys = 1e6
                value = float(clip(value, min_phys, max_phys))

            # abnormal flag if outside normal range
            abnormal = 1 if (value < low or value > high) else 0
            if abnormal:
                abnormal_lab_count += 1
            cur.execute(
                "INSERT INTO LabResults (admission_id, patient_id, test_name, test_code, value, unit, normal_low, normal_high, abnormal, date) VALUES (?,?,?,?,?,?,?,?,?,?);",
                (admission_id, patient_id, lab_name, info['code'], value, info['unit'], low, high, abnormal, admission_date_str)
            )
            lab_rows += 1

        # Investigations/Imaging: 0-2 imaging per admission (20% none, 50% one, 30% two)
        r = random.random()
        if r < 0.20:
            num_imaging = 0
        elif r < 0.70:
            num_imaging = 1
        else:
            num_imaging = 2

        imaging_abnormal_count = 0
        for _ in range(num_imaging):
            code, name = random.choice(IMAGING_TYPES)
            # Determine abnormal probability
            if diagnosis in ['Ischemic stroke', 'Hemorrhagic stroke', 'Brain tumor']:
                prob_abn = 0.60
            else:
                prob_abn = 0.15
            abnormal = 1 if random.random() < prob_abn else 0
            result_text = 'abnormal' if abnormal else 'normal'
            # Look up InvestigationTypes.type_id for the imaging code (insert if missing per spec)
            cur.execute("SELECT type_id FROM InvestigationTypes WHERE code = ?;", (code,))
            row = cur.fetchone()
            if row:
                type_id = row[0]
            else:
                cur.execute("INSERT INTO InvestigationTypes (code, name, category) VALUES (?,?,?);", (code, name, 'imaging'))
                type_id = cur.lastrowid
            cur.execute(
                "INSERT INTO Investigations (admission_id, patient_id, type_id, result_text, result_value, unit, abnormal, date) VALUES (?,?,?,?,?,?,?,?);",
                (admission_id, patient_id, type_id, result_text, None, None, abnormal, admission_date_str)
            )
            inv_rows += 1
            if abnormal:
                imaging_abnormal_count += 1

        # Additionally store GCS and STOPBANG as Investigations using InvestigationTypes mapping
        # GCS store as result_value
        cur.execute("SELECT type_id FROM InvestigationTypes WHERE code = 'GCS';")
        gcs_row = cur.fetchone()
        if gcs_row:
            gcs_type_id = gcs_row[0]
        else:
            cur.execute("INSERT INTO InvestigationTypes (code, name, category) VALUES (?,?,?);", ('GCS','GCS','scale'))
            gcs_type_id = cur.lastrowid
        cur.execute("INSERT INTO Investigations (admission_id, patient_id, type_id, result_text, result_value, unit, abnormal, date) VALUES (?,?,?,?,?,?,?,?);",
                    (admission_id, patient_id, gcs_type_id, None, gcs_total, 'points', 0, admission_date_str))
        inv_rows += 1

        # STOPBANG
        cur.execute("SELECT type_id FROM InvestigationTypes WHERE code = 'STOPBANG';")
        stop_row = cur.fetchone()
        if stop_row:
            stop_type_id = stop_row[0]
        else:
            cur.execute("INSERT INTO InvestigationTypes (code, name, category) VALUES (?,?,?);", ('STOPBANG','STOPBANG','scale'))
            stop_type_id = cur.lastrowid
        cur.execute("INSERT INTO Investigations (admission_id, patient_id, type_id, result_text, result_value, unit, abnormal, date) VALUES (?,?,?,?,?,?,?,?);",
                    (admission_id, patient_id, stop_type_id, None, stopbang_score, 'points', 0, admission_date_str))
        inv_rows += 1

        # PatientMedication assignment: correlate with comorbidity_count and severity_score
        lam = 1.0 + comorbidity_count * 0.8 + (severity_score / 10.0)
        num_meds = int(clip(np.random.poisson(lam), 0, 6))
        meds_assigned = random.sample(meds_all, k=min(len(meds_all), num_meds)) if num_meds > 0 else []
        for med_id in meds_assigned:
            # Insert OR IGNORE to avoid duplicate (patient,med) pairs across admissions
            cur.execute("INSERT OR IGNORE INTO PatientMedication (patient_id, medicine_id, current_medication) VALUES (?,?,?);", (patient_id, med_id, 1))
            med_assignments += 1


        cur.execute("SELECT Diagnosis FROM Patients WHERE UID = ?;", (patient_id,))
        r = cur.fetchone()
        diagnosis = r[0]

        # Save metadata for this admission for later PatientFeatures generation
        admissions_created.append({
            'admission_id': admission_id,
            'patient_id': patient_id,
            'admission_date': admission_date_str,
            'discharge_date': discharge_date_str,
            'status': status,
            'diagnosis': diagnosis,
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'neck': neck,
            'gcs_total': gcs_total,
            'wfns_grade': wfns_grade,
            'stopbang_score': stopbang_score,
            'severity_score': severity_score,
            'comorbidity_count': comorbidity_count,
            'imaging_abnormal_count': imaging_abnormal_count,
            'actual_recovery_days': actual_recovery_days
        })

    # Commit large transaction
    conn.commit()

    # Print counts of inserted rows for feedback
    print(f"Inserted: Lab rows ~{lab_rows}, Investigation rows ~{inv_rows}, NeurologicalScales ~{scales_rows}")
    print(f"Comorbidity rows {comorb_rows}, RiskFactors {risk_rows}, PreopChecklist {preop_rows}, Medication assignments {med_assignments}")

    return admissions_created


def build_patient_features(conn: sqlite3.Connection, admissions_created):
    """Build a PatientFeatures row for each discharged admission only (training examples).

    Uses most recent lab before discharge (we only stored labs on admission date, so uses those).
    Computes severity_score deterministically per the explicit formula in spec.
    """
    print("Generating PatientFeatures for discharged admissions...")
    cur = conn.cursor()
    features_inserted = 0

    for adm in admissions_created:
        if adm['status'] != 'discharged':
            continue
        admission_id = adm['admission_id']
        patient_id = adm['patient_id']
        age = adm['age']
        sex = adm['sex']
        bmi = adm['bmi']
        diagnosis = adm['diagnosis']
        blood_type_row = cur.execute("SELECT BloodType FROM Patients WHERE UID = ?;", (patient_id,)).fetchone()
        blood_type = blood_type_row[0] if blood_type_row else None
        gcs_total = adm['gcs_total']
        wfns_grade = adm['wfns_grade']
        stopbang_score = adm['stopbang_score']

        # For each key lab, pick most recent lab (we only have admission-date labs, so find any lab for admission)
        lab_values = {}
        abnormal_lab_count = 0
        for lab_name in ['Sodium','Potassium','Creatinine','GFR','ALT','AST','Bilirubin','Hemoglobin','WBC','Platelets','Blood_Sugar']:
            cur.execute("SELECT value, abnormal FROM LabResults WHERE admission_id = ? AND test_name = ? ORDER BY date DESC LIMIT 1;", (admission_id, lab_name))
            lr = cur.fetchone()
            if lr:
                val = lr[0]
                abnormal = lr[1]
                lab_values[lab_name] = val
                if abnormal == 1:
                    abnormal_lab_count += 1
            else:
                lab_values[lab_name] = None

        # num_medications: count PatientMedication rows for this patient
        cur.execute("SELECT COUNT(*) FROM PatientMedication WHERE patient_id = ?;", (patient_id,))
        num_medications = cur.fetchone()[0]

        # num_investigations: count Investigations for this admission
        cur.execute("SELECT COUNT(*) FROM Investigations WHERE admission_id = ?;", (admission_id,))
        num_investigations = cur.fetchone()[0]

        # imaging_abnormal_count: count Investigations for this admission with abnormal=1 and whose type is imaging
        cur.execute("SELECT COUNT(1) FROM Investigations inv JOIN InvestigationTypes it ON inv.type_id = it.type_id WHERE inv.admission_id = ? AND inv.abnormal = 1 AND it.category = 'imaging';", (admission_id,))
        imaging_abnormal_count = cur.fetchone()[0]

        # comorbidity_count
        cur.execute("SELECT SUM(present) FROM Comorbidities WHERE admission_id = ?;", (admission_id,))
        ccount_row = cur.fetchone()
        comorbidity_count = int(ccount_row[0]) if ccount_row and ccount_row[0] is not None else 0

        # severity_score formula (explicit)
        age_factor = int(max(0, (age - 60) / 20))
        if gcs_total is not None:
            severity_score_comp = int(round((20 - gcs_total) + 0.8 * comorbidity_count + 0.5 * abnormal_lab_count + age_factor))
            severity_score_comp = int(clip(severity_score_comp, 1, 20))
        else:
            # fallback: try to use stored admission severity or default
            severity_score_comp = adm.get('severity_score') or int(clip(int(round(np.random.normal(5,2))),1,20))

        # Compute recovery_days as days between discharge_date and admission_date (must be integer >=1)
        admission_dt = datetime.strptime(adm['admission_date'], '%Y-%m-%d %H:%M:%S')
        discharge_dt = datetime.strptime(adm['discharge_date'], '%Y-%m-%d %H:%M:%S')
        recovery_days = (discharge_dt - admission_dt).days
        if recovery_days < 1:
            recovery_days = 1

        # Insert PatientFeatures row
        cur.execute(
            "INSERT INTO PatientFeatures (admission_id, patient_id, age, sex, bmi, diagnosis, blood_type, gcs_total, wfns_grade, stopbang_score, sodium, potassium, creatinine, gfr, alt, ast, bilirubin, hemoglobin, wbc, platelets, blood_sugar, num_medications, num_investigations, imaging_abnormal_count, comorbidity_count, severity_score, admission_date, discharge_date, recovery_days) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);",
            (
                admission_id, patient_id, age, sex, bmi, diagnosis, blood_type, gcs_total, wfns_grade, stopbang_score,
                lab_values.get('Sodium'), lab_values.get('Potassium'), lab_values.get('Creatinine'), lab_values.get('GFR'), lab_values.get('ALT'), lab_values.get('AST'), lab_values.get('Bilirubin'), lab_values.get('Hemoglobin'), lab_values.get('WBC'), lab_values.get('Platelets'), lab_values.get('Blood_Sugar'),
                num_medications, num_investigations, imaging_abnormal_count, comorbidity_count, severity_score_comp,
                adm['admission_date'], adm['discharge_date'], recovery_days
            )
        )
        features_inserted += 1

    conn.commit()
    print(f"Inserted PatientFeatures rows: {features_inserted}")
    return features_inserted


def validate_db_and_print(conn: sqlite3.Connection, db_path: str, start_date: datetime, end_date: datetime):
    """Run validations and print required summary outputs as specified.

    The spec lists a series of checks; this function implements them and prints the results.
    """
    print("\n--- Validation & summary checks ---")
    cur = conn.cursor()

    # 1. Counts: Patients==1000, Beds==100, Medications==30
    cur.execute("SELECT COUNT(*) FROM Patients;")
    patients_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM Beds;")
    beds_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM Medications;")
    meds_count = cur.fetchone()[0]

    print(f"Patients: {patients_count}")
    print(f"Beds: {beds_count}")
    print(f"Medications: {meds_count}")

    # 2. Admissions: total and % discharged
    cur.execute("SELECT COUNT(*) FROM Admissions;")
    admissions_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM Admissions WHERE status = 'discharged';")
    admissions_discharged = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM Admissions WHERE status = 'admitted';")
    admissions_admitted = cur.fetchone()[0]
    discharged_pct = (admissions_discharged / admissions_total * 100) if admissions_total else 0

    print(f"Admissions: {admissions_total} (discharged {discharged_pct:.1f}%, admitted {100-discharged_pct:.1f}%)")

    # 3. PatientFeatures rows equal to number of discharged admissions
    cur.execute("SELECT COUNT(*) FROM PatientFeatures;")
    pf_count = cur.fetchone()[0]
    print(f"PatientFeatures (discharged): {pf_count} (should equal discharged admissions {admissions_discharged})")

    # 4. No discharge dates after now
    cur.execute("SELECT admission_id, discharge_date FROM Admissions WHERE discharge_date IS NOT NULL AND datetime(discharge_date) > datetime('now');")
    future_discharges = cur.fetchall()
    if future_discharges:
        print("Found admissions with discharge_date in the future. Re-classifying them as 'admitted' and clearing discharge_date...")
        for row in future_discharges:
            aid = row[0]
            print(f" - Fixing admission_id {aid}")
            cur.execute("UPDATE Admissions SET status = 'admitted', discharge_date = NULL WHERE admission_id = ?;", (aid,))
        conn.commit()
    else:
        print("No discharge dates after now found.")

    # 5. No pregnancy flagged for males; no pregnancy for females outside 12..50
    cur.execute("SELECT rf.factor_id, rf.admission_id, rf.patient_id, rf.pregnancy FROM RiskFactors rf JOIN Patients p ON rf.patient_id = p.UID WHERE rf.pregnancy = 1 AND p.Sex = 'M';")
    bad_preg_males = cur.fetchall()
    if bad_preg_males:
        print("Found pregnancy flagged for males; fixing...")
        for r in bad_preg_males:
            fid = r[0]
            print(f" - Clearing pregnancy for factor_id {fid}")
            cur.execute("UPDATE RiskFactors SET pregnancy = 0 WHERE factor_id = ?;", (fid,))
        conn.commit()
    cur.execute("SELECT rf.factor_id, rf.admission_id, rf.patient_id, rf.pregnancy, p.Age FROM RiskFactors rf JOIN Patients p ON rf.patient_id = p.UID WHERE rf.pregnancy = 1 AND (p.Age < 12 OR p.Age > 50);")
    bad_preg_ages = cur.fetchall()
    if bad_preg_ages:
        print("Found pregnancy flagged for females outside 12..50; fixing...")
        for r in bad_preg_ages:
            fid = r[0]
            print(f" - Clearing pregnancy for factor_id {fid} (patient age {r[4]})")
            cur.execute("UPDATE RiskFactors SET pregnancy = 0 WHERE factor_id = ?;", (fid,))
        conn.commit()
    else:
        print("Pregnancy flags validated.")

    # 6. Print sample PatientFeatures rows ORDER BY RANDOM() LIMIT 5
    print("\nSample PatientFeatures rows (up to 5):")
    cur.execute("SELECT * FROM PatientFeatures ORDER BY RANDOM() LIMIT 5;")
    rows = cur.fetchall()
    col_names = [d[0] for d in cur.description] if cur.description else []
    # Print header then rows in CSV-like format
    if col_names:
        print(','.join(col_names))
    for r in rows:
        print(','.join([str(x) if x is not None else '' for x in r]))

    # 7. Recovery_days stats
    cur.execute("SELECT MIN(recovery_days), CAST((SELECT value FROM (SELECT recovery_days as value FROM PatientFeatures ORDER BY recovery_days LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM PatientFeatures)) ) AS INTEGER), AVG(recovery_days) FROM PatientFeatures;")
    stats_min_med_mean = cur.fetchone()
    # Better compute median and 90th percentile in Python for accuracy
    cur.execute("SELECT recovery_days FROM PatientFeatures WHERE recovery_days IS NOT NULL ORDER BY recovery_days;")
    recovery_days_list = [r[0] for r in cur.fetchall()]
    if recovery_days_list:
        recovery_arr = np.array(recovery_days_list)
        mn = int(np.min(recovery_arr))
        median = int(np.median(recovery_arr))
        mean = float(np.mean(recovery_arr))
        p90 = int(np.percentile(recovery_arr,90))
    else:
        mn = median = mean = p90 = None

    mean_str = f"{mean:.2f}" if mean is not None else "N/A"
    print(f"Recovery days stats: min={mn}, median={median}, mean={mean_str}, 90th={p90}")

    # Print counts of LabResults and Investigations
    cur.execute("SELECT COUNT(*) FROM LabResults;")
    lab_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM Investigations;")
    inv_count = cur.fetchone()[0]

    print(f"Lab rows: {lab_count}, Investigations rows: {inv_count}, PatientFeatures (discharged): {pf_count}")

    # Final DB path and date range
    print(f"Database created at: {db_path}")
    print(f"Random seed = {SEED} and Date range = [{start_date.strftime('%Y-%m-%d')}] to [{end_date.strftime('%Y-%m-%d')}]")

    # Return some of the computed values for printing by caller if needed
    return {
        'patients': patients_count,
        'beds': beds_count,
        'medications': meds_count,
        'admissions_total': admissions_total,
        'admissions_discharged': admissions_discharged,
        'admissions_admitted': admissions_admitted,
        'lab_rows': lab_count,
        'investigations_rows': inv_count,
        'patientfeatures': pf_count,
        'recovery_stats': {'min': mn, 'median': median, 'mean': mean, '90th': p90}
    }


def export_tables_to_csv(conn: sqlite3.Connection, out_dir: str):
    """Export each table to CSV files in out_dir. This is optional and triggered by a flag.
    Returns list of file paths written.
    """
    print(f"Exporting tables to CSV in {out_dir} ...")
    os.makedirs(out_dir, exist_ok=True)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    paths = []
    for t in tables:
        path = os.path.join(out_dir, f"{t.lower()}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cur2 = conn.cursor()
            cur2.execute(f"SELECT * FROM {t};")
            cols = [d[0] for d in cur2.description]
            writer.writerow(cols)
            for row in cur2.fetchall():
                writer.writerow(row)
        paths.append(path)
    print(f"Wrote {len(paths)} CSV files.")
    return paths


# ----------------------
# Main entry
# ----------------------

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic neurology SQLite dataset')
    parser.add_argument('--db-path', type=str, default=DB_DEFAULT_PATH, help='Path to output SQLite database')
    parser.add_argument('--append', action='store_true', help='If set, append to existing DB instead of removing it')
    parser.add_argument('--export-csv', action='store_true', default=EXPORT_CSV_DEFAULT, help='Export tables to CSV files in same folder')
    args = parser.parse_args()

    db_path = args.db_path
    print(os.path.isfile(db_path))
    start_date = datetime.now() - timedelta(days=365*YEARS)
    end_date = datetime.now()

    # Remove existing DB unless append requested
    if os.path.isfile(db_path) and not args.append:
        print(f"Removing existing DB at {db_path} for reproducible run...")
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"Could not remove {db_path}: {e}")
            sys.exit(1)
    else:
        if args.append:
            print("Append mode: existing DB will be used (if present).")

    db_dir = os.path.dirname(db_path)
    if db_dir:  # only create if there is a directory in the path
        # Ensure directory exists
        os.makedirs(db_dir, exist_ok=True)


    try:
        conn = sqlite3.connect(db_path)
        create_schema(conn)
        populate_medications(conn)
        populate_investigation_and_imaging_types(conn)
        populate_beds(conn)
        patients = generate_patients(conn)
        admissions_created = generate_admissions_and_related(conn, patients)
        features_count = build_patient_features(conn, admissions_created)

        # Run validations & print summary – this also prints sample PatientFeatures rows
        summary = validate_db_and_print(conn, db_path, start_date, end_date)

        # Optionally export CSVs
        csv_paths = []
        if args.export_csv:
            csv_paths = export_tables_to_csv(conn, os.path.dirname(db_path))

        # Final printed summary per spec (some lines already printed in validate_db)
        print("\n--- Final summary (concise) ---")
        print(f"Database created at: {db_path}")
        print(f"Patients: {summary['patients']}")
        print(f"Admissions: {summary['admissions_total']} (discharged {summary['admissions_discharged'] / summary['admissions_total'] * 100:.1f}%, admitted {summary['admissions_admitted'] / summary['admissions_total'] * 100:.1f}%)")
        print(f"Beds: {summary['beds']}, Medications: {summary['medications']}")
        print(f"Lab rows: {summary['lab_rows']}, Investigations rows: {summary['investigations_rows']}, PatientFeatures (discharged): {summary['patientfeatures']}")
        rs = summary['recovery_stats']
        mean_str = f"{rs['mean']:.2f}" if rs['mean'] is not None else "N/A"
        print(f"Recovery days stats: min={rs['min']}, median={rs['median']}, mean={mean_str}, 90th={rs['90th']}")

        print(f"Random seed = {SEED}")
        print(f"Date range = [{start_date.strftime('%Y-%m-%d')}] to [{end_date.strftime('%Y-%m-%d')}]")
        if csv_paths:
            print("CSV exports:")
            for p in csv_paths:
                print(f" - {p}")

        # Explanation of synthetic relationships and where noise/missingness applied
        print("\nShort explanation of synthetic relationships and noise:")
        print(" - Severity is correlated with diagnosis, age, GCS and comorbidity count. Higher severity tends to increase recovery_days.")
        print(" - Labs were sampled from normal ranges with 15% missingness and measurement noise (5% SD). High severity biases select labs toward abnormal values (e.g., creatinine up, GFR down, WBC up).")
        print(" - 2% of admissions were injected with long-stay outliers (+10-60 days) to simulate rare prolonged admissions.")
        print(" - Imaging abnormality probability is higher for stroke and tumor diagnoses.")
        print(" - PatientMedication counts correlate with comorbidity_count and severity via a Poisson model.")

    except Exception as e:
        print("An error occurred during generation:")
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except:
            pass


if __name__ == '__main__':
    main()