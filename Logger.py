import sqlite3, serial
from datetime import datetime

# serial connection
ser = serial.Serial('COM5', 115200) 

# Connect to SQLite
conn = sqlite3.connect("CU_Nexus_Ai/RFID.db")
cursor = conn.cursor()



# cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
# conn.commit()
# conn.close()

print("Listening for RFID tags...")

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith("UID:"):
            data = line.replace("UID:", "").strip()
            parts = data.split(" ON_BED:")

            uid = parts[0]
            on_bed = parts[1]
            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("Select Bed_id FROM Patients WHERE UID = ?;", (uid,))
            bed_id = cursor.fetchone()
            bed_id = bed_id[0]

            print(f"onbed: {on_bed}")
            if (bed_id != None):
                if(int(on_bed) == 1):
                    cursor.execute("UPDATE Beds SET patient_id = ? WHERE Bed_id = ?;", (uid, bed_id))
                    print("occupied")
                else:
                    cursor.execute("UPDATE Beds SET patient_id = NULL WHERE Bed_id = ?;", (bed_id,))
                    print("not occupied")
                # Insert into database
                cursor.execute("INSERT INTO BedLogs (Bed_id, patient_id, On_bed, log_time) VALUES (?, ?, ?, ?)", (bed_id, uid, on_bed, log_time))
                conn.commit()
                print(f"Logged: UID={uid}, Time={log_time}, onbed={on_bed}, bed_id={bed_id}")
                
except KeyboardInterrupt:
    print("Exiting...")
    conn.close()
    ser.close()