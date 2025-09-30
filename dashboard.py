import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import joblib
import os
from dotenv import load_dotenv
import pickle
from datetime import timedelta
import re

load_dotenv()
API_KEY = os.getenv("GOOGLE_API")

if not API_KEY:
    raise RuntimeError("API_KEY not set — copy .env.example to .env and add your key")

genai.configure(api_key=API_KEY)

DB_PATH = "RFID.db"

def run_query(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# ------------------------
# Classic ML Model Placeholder
# ------------------------
model = joblib.load("model.pkl")

def predict_recovery(admission_id : int):
    conn = sqlite3.connect("RFID.db")
    row = pd.read_sql(f"SELECT * FROM PatientFeatures WHERE admission_id={admission_id}", conn)

    if not row.empty:
        # drop leakage cols
        drop_cols = ["feature_id","admission_id","patient_id","recovery_days",
                        "discharge_date","predicted_recovery_days","predicted_discharge_date",
                        "model_version","prediction_confidence","created_at"]
        X_new = row.drop(columns=[c for c in drop_cols if c in row.columns])

        pred_days = int(model.predict(X_new)[0])
        admission_date = pd.to_datetime(row["admission_date"][0])
        pred_discharge = (admission_date + timedelta(days=pred_days)).strftime("%Y-%m-%d")
        print(pred_days)
        print(pred_discharge)
        print(admission_date)

        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO Predictions (admission_id, patient_id, predicted_recovery_days,
                                predicted_discharge_date, model_version, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (int(row["admission_id"]), row["patient_id"][0],
            pred_days, pred_discharge, "RF_v1.0", 0.85))
        conn.commit()
        print(f"Prediction inserted: {pred_days} days, discharge {pred_discharge}")
    return pred_days



def safe_sql(sql: str) -> bool:
    """Allow only SELECT statements to prevent modification of DB"""
    return sql.strip().lower().startswith("select")


# ------------------------
# LLM Chat Function
# ------------------------
def chat_with_llm(user_input: str):
    model = genai.GenerativeModel("gemini-2.5-flash")
    admission_match = re.search(r"(?:admission[_ ]?id[:=]?\s*)(\w+)", user_input, re.IGNORECASE)
    admission_id = admission_match.group(1) if admission_match else None

    pred_info = ""
    pred_days = 0
    if admission_id:
        # --- Step 2: Get patient_id for this admission ---
        df = run_query("SELECT patient_id FROM Admissions WHERE admission_id = ?", (admission_id,))
        if not df.empty:
            patient_id = df["patient_id"].iloc[0]

            # --- Step 3: Run ML prediction ---
            pred_days = predict_recovery(admission_id)
            pred_df = run_query("SELECT * FROM Predictions WHERE admission_id = ? ORDER BY rowid DESC LIMIT 1", (admission_id,))
            pred_info = f"Prediction for admission {admission_id}:\n{pred_df.to_string(index=False)}\n"


    patients = run_query("SELECT * FROM Patients LIMIT 5;").to_string()
    beds = run_query("SELECT * FROM Beds LIMIT 5;").to_string()
    admissions = run_query("SELECT * FROM Admissions LIMIT 5;").to_string()
    investigations = run_query("SELECT * FROM Investigations LIMIT 5;").to_string()
    lab_results = run_query("SELECT * FROM LabResults LIMIT 5;").to_string()
    neuro_scales = run_query("SELECT * FROM NeurologicalScales LIMIT 5;").to_string()

    schema = """ Admissions(admission_id, patient_id, bed_id, admission_date, discharge_date,admission_reason, diagnosis, status, primary_surgeon, preop_done)\n
        DiagnosisCodes(code, description)
        InvestigationTypes(type_id, code, name, category)
        ImagingTypes(imaging_type_id, code, name)
        Investigations(investigation_id, admission_id, patient_id, type_id,result_text, result_value, unit, abnormal, date)
        LabResults(lab_id, admission_id, patient_id, test_name, test_code, value, unit, normal_low, normal_high, abnormal, date)
        NeurologicalScales(scale_id, admission_id, patient_id, gcs_eye, gcs_verbal, gcs_motor, gcs_total, wfns_grade, stop_snore, stop_tired, stop_observed, stop_bp, stop_bmi, stop_age, stop_neck, stop_gender, stopbang_score, date)
        Comorbidities(patient_id, admission_id, comorbidity, present)
        RiskFactors(factor_id, admission_id, patient_id, bmi, neck_circumference, smoker, pregnancy, allergy_contrast, airway_difficulty, date)
        PreoperativeChecklist(checklist_id, admission_id, patient_id, bp_systolic, bp_diastolic, ecg_abnormal, airway_assessment, pregnancy_test, coag_pt, coag_inr, coag_aptt, blood_sugar, date)
        PatientFeatures(feature_id, admission_id, patient_id, age, sex, bmi,
                        blood_type, gcs_total, wfns_grade, stopbang_score,
                        sodium, potassium, creatinine, gfr, alt, ast, bilirubin,
                        hemoglobin, wbc, platelets, blood_sugar,
                        num_medications, num_investigations, comorbidity_count,
                        severity_score, admission_date, discharge_date, recovery_days,
                        predicted_recovery_days, prediction_timestamp,
                        model_version, prediction_confidence, created_at)
        Predictions(prediction_id, admission_id, patient_id,
                    predicted_recovery_days, predicted_discharge_date,
                    model_version, confidence, created_at)"""
    

    sqlPrompt = f"""You are a SQL expert.
    The database schema is:\n{schema}

    User asked: {user_input}

    If relevant, include Predictions in your query.
    Write ONE SQL SELECT query that best answers this question.
    Output ONLY the SQL query inside triple backticks.
    """""


    sql_response = model.generate_content(sqlPrompt).text
    sql_match = re.search(r"```(?:sql)?\s*(.*?)```", sql_response, re.DOTALL | re.IGNORECASE)

    sql_query = None
    results = None
    if sql_match:
        sql_query = sql_match.group(1).strip()
        if safe_sql(sql_query):
            try:
                results = run_query(sql_query)
            except Exception as e:
                results = pd.DataFrame({"Error": [str(e)]})
        else:
            results = pd.DataFrame({"Error": ["Unsafe SQL query blocked"]})

    context = f"""
    Database snapshot:
    Patients:\n{patients}\n
    Beds:\n{beds}\n
    Admissions:\n{admissions}\n
    Investigations:\n{investigations}\n
    Lab Results:\n{lab_results}\n
    Neurological Scales:\n{neuro_scales}\n

       
    User input: {user_input}

    SQL Query:
    {sql_query if sql_query else "None"}

    Query Results:
    {results.to_string() if results is not None and not results.empty else "None"}

    ML Prediction:\n{pred_days}\n
    more prediction info: \n{pred_info}\n
    """

    prompt = f"""
    You are a hospital assistant AI.
    User asked: {user_input}

    Here is the structured data:\n{context}

    - Provide a clear, professional answer.
    - Do not invent data not present in the database.
    - Use this data to answer the question in a clear, professional way.
    - If the user asked about an admission_id, include the latest ML prediction.
    - Otherwise, answer using relevant database information.
    - If nothing relevant is found, say so politely.
    """

    response = model.generate_content(prompt)
    if response.candidates and response.candidates[0].content.parts:
        sql_response = "".join(part.text for part in response.candidates[0].content.parts if part.text)
    else:
        sql_response = "sth went wrong"   # or None, or some fallback message
    return response.text



# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Hospital Dashboard", layout="wide")

st.title("🏥 Hospital Bed & Patient Management")

tabs = st.tabs(["Beds", "Patients", "Add/Edit Patient", "Chat Assistant"])

# ------------------------
# Beds Tab
# ------------------------
with tabs[0]:
    st.subheader("Bed Status")
    beds_df = run_query("SELECT * FROM Beds;")
    st.dataframe(beds_df)

# ------------------------
# Patients Tab
# ------------------------
with tabs[1]:
    patients_df = run_query("SELECT * FROM Patients;")


    if not patients_df.empty:
        patient_id = st.selectbox(
            "Select Patient",
            patients_df["UID"],  # values returned
            format_func=lambda uid: patients_df.loc[patients_df["UID"] == uid, "Name"].values[0]  # what to display
        )
    else:
        patient_id = None


    # Only fetch details if a patient is selected
    if patient_id:
        st.write("### Investigations")
        inv_df = run_query("SELECT * FROM Investigations WHERE patient_id = ?", (patient_id,))
        st.dataframe(inv_df)

        st.write("### Medications")
        meds_df = run_query("""
            SELECT M.medicine_name, PM.current_medication
            FROM PatientMedication PM
            JOIN Medications M ON PM.medicine_id = M.medicine_id
            WHERE PM.patient_id = ?;
        """, (patient_id,))
        st.dataframe(meds_df)


    st.subheader("Patients")
    st.dataframe(patients_df)

# ------------------------
# Add/Edit Patient Tab
# ------------------------
with tabs[2]:
    st.subheader("➕➖ Add / Edit / Delete Patient")

    # --- ADD NEW PATIENT ---
    with st.expander("Add New Patient"):
        with st.form("add_patient_form"):
            uid = st.text_input("UID")
            name = st.text_input("Name")
            phone = st.text_input("Phone")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            bloodtype = st.text_input("Blood Type")
            admission_reason = st.text_input("Admission Reason")
            family_history = st.text_input("Family History of Disease")
            diagnosis = st.text_input("Diagnosis")
            expected_recovery = st.text_input("Expected Recovery")

            submitted = st.form_submit_button("Add Patient")
            if submitted:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO Patients (UID, Name, Phone, Age, BloodType, Admission_reason,
                                              Family_history_disease, Diagnosis, Expected_Recovery)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (uid, name, phone, age, bloodtype, admission_reason,
                          family_history, diagnosis, expected_recovery))
                    conn.commit()
                    conn.close()
                    st.success(f"✅ Patient {name} added successfully!")
                except sqlite3.IntegrityError:
                    st.error("❌ UID already exists. Please choose another.")

    # --- EDIT EXISTING PATIENT ---
    with st.expander("Edit Existing Patient"):
        patients_df = run_query("SELECT UID, Name FROM Patients;")
        if not patients_df.empty:
            selected_uid = st.selectbox("Select Patient to Edit", patients_df["UID"])
            if selected_uid:
                patient_data = run_query("SELECT * FROM Patients WHERE UID = ?", (selected_uid,))
                if not patient_data.empty:
                    patient = patient_data.iloc[0]

                    with st.form("edit_patient_form"):
                        name = st.text_input("Name", patient["Name"])
                        phone = st.text_input("Phone", patient["Phone"])
                        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=patient["Age"])
                        bloodtype = st.text_input("Blood Type", patient["BloodType"])
                        admission_reason = st.text_input("Admission Reason", patient["Admission_reason"])
                        family_history = st.text_input("Family History", patient["Family_history_disease"])
                        diagnosis = st.text_input("Diagnosis", patient["Diagnosis"])
                        expected_recovery = st.text_input("Expected Recovery", patient["Expected_Recovery"])

                        update_btn = st.form_submit_button("Update Patient")
                        if update_btn:
                            conn = sqlite3.connect(DB_PATH)
                            cur = conn.cursor()
                            cur.execute("""
                                UPDATE Patients
                                SET Name=?, Phone=?, Age=?, BloodType=?, Admission_reason=?,
                                    Family_history_disease=?, Diagnosis=?, Expected_Recovery=?
                                WHERE UID=?
                            """, (name, phone, age, bloodtype, admission_reason,
                                  family_history, diagnosis, expected_recovery, selected_uid))
                            conn.commit()
                            conn.close()
                            st.success(f"✅ Patient {selected_uid} updated successfully!")

    # --- DELETE PATIENT ---
    with st.expander("Delete Patient"):
        patients_df = run_query("SELECT UID, Name FROM Patients;")
        if not patients_df.empty:
            delete_uid = st.selectbox("Select Patient to Delete", patients_df["UID"])
            if st.button("Delete Patient"):
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("DELETE FROM Patients WHERE UID = ?", (delete_uid,))
                conn.commit()
                conn.close()
                st.warning(f"🗑️ Patient {delete_uid} deleted successfully!")


# ------------------------
# Chat Assistant Tab
# ------------------------
with tabs[3]:
    st.subheader("💬 AI Chat Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question...")
    if st.button("Send") and user_input:
        reply = chat_with_llm(user_input)
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("AI", reply))

    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")
