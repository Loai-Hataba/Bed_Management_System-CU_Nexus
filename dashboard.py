import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()                   # loads .env from project root
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not set — copy .env.example to .env and add your key")

# Use safely — don't print the full key
masked = API_KEY[:4] + "..." + API_KEY[-4:] if len(API_KEY) > 8 else "****"
print("API key loaded (masked):", masked)


# CONFIG: Gemini API Key
genai.configure(api_key="")

# DB Helper Functions
DB_PATH = "CU_Nexus_Ai/RFID.db"

def run_query(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# ------------------------
# Classic ML Model Placeholder
# FIXME:
# ------------------------
def predict_recovery(patient_id: str):
    # TODO: Replace with your real ML model
    # For demo: fake prediction
    return f"Patient {patient_id} is expected to recover in ~7 days."

# ------------------------
# LLM Chat Function
# FIXME:
# ------------------------
def chat_with_llm(user_input: str):
    # Always fetch DB info (simplified example: patients + beds)
    patients = run_query("SELECT * FROM Patients LIMIT 5;").to_string()
    beds = run_query("SELECT * FROM Beds LIMIT 5;").to_string()

    # Always run ML prediction for demo (on 1 patient)
    ml_pred = predict_recovery("P001")

    # Prepare context for Gemini
    context = f"""
    Database snapshot:
    Patients:\n{patients}\n
    Beds:\n{beds}\n

    ML Prediction:\n{ml_pred}\n
    """

    prompt = f"""
    You are a hospital assistant AI.
    User asked: {user_input}

    Here is the structured data:\n{context}

    Use this data to answer the question in a clear, professional way.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
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
    st.subheader("Patients")
    patients_df = run_query("SELECT * FROM Patients;")
    st.dataframe(patients_df)

    # Click to view details
    patient_id = st.selectbox("Select Patient", patients_df["UID"] if not patients_df.empty else [])
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
            bed_id = st.number_input("Assigned Bed ID", min_value=1, step=1)

            submitted = st.form_submit_button("Add Patient")
            if submitted:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO Patients (UID, Name, Phone, Age, BloodType, Admission_reason,
                                              Family_history_disease, Diagnosis, Expected_Recovery, Bed_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (uid, name, phone, age, bloodtype, admission_reason,
                          family_history, diagnosis, expected_recovery, bed_id))
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
                        bed_id = st.number_input("Assigned Bed ID", min_value=1, step=1, value=patient["Bed_id"])

                        update_btn = st.form_submit_button("Update Patient")
                        if update_btn:
                            conn = sqlite3.connect(DB_PATH)
                            cur = conn.cursor()
                            cur.execute("""
                                UPDATE Patients
                                SET Name=?, Phone=?, Age=?, BloodType=?, Admission_reason=?,
                                    Family_history_disease=?, Diagnosis=?, Expected_Recovery=?, Bed_id=?
                                WHERE UID=?
                            """, (name, phone, age, bloodtype, admission_reason,
                                  family_history, diagnosis, expected_recovery, bed_id, selected_uid))
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
# FIXME:
# ------------------------
# with tabs[3]:
#     st.subheader("💬 AI Chat Assistant")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     user_input = st.text_input("Ask a question...")
#     if st.button("Send") and user_input:
#         reply = chat_with_llm(user_input)
#         st.session_state.chat_history.append(("User", user_input))
#         st.session_state.chat_history.append(("AI", reply))

#     for role, msg in st.session_state.chat_history:
#         st.markdown(f"**{role}:** {msg}")
