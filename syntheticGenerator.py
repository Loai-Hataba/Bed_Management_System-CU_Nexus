# synthetic_bedlogs.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(0)
np.random.seed(0)

PATIENT_IDS = [
    "A38FF70E","DAB1C5B2","04537934230289","C41E9A7B","9FD2B608","E73AB4C1",
    "5A8CF3D2","B9E41F07","7D32ACFE","4F0B89C3","D6172BAE","8C5FD940","FA31B62D"
]

def generate_bed_logs(num_beds=13, days=7, freq_minutes=60, horizon_hours=4):
    rows = []
    start = datetime.now() - timedelta(days=days)
    total_steps = int((days * 24 * 60) / freq_minutes)
    times = [start + timedelta(minutes=i * freq_minutes) for i in range(total_steps)]

    horizon_steps = horizon_hours * 60 // freq_minutes

    for bed in range(1, num_beds + 1):
        occupied = 0
        prev_occupied = occupied
        current_patient = random.choice(PATIENT_IDS)

        last_change_time = start
        exits_history = []

        bed_rows = []

        for t in times:
            hour = t.hour
            base_prob = 0.4 + 0.4 * (8 <= hour <= 20)
            base_prob += 0.1 * (bed % 3 == 0)

            if random.random() < 0.02:
                occupied = 1 - occupied
            if random.random() < 0.01:
                occupied = 1 - occupied

            if prev_occupied != occupied:
                last_change_time = t
                if prev_occupied == 1 and occupied == 0:
                    exits_history.append(t)

            if prev_occupied == 0 and occupied == 1:
                choices = [p for p in PATIENT_IDS if p != current_patient]
                current_patient = random.choice(choices) if choices else random.choice(PATIENT_IDS)

            # features
            time_since_change = int((t - last_change_time).total_seconds() // 60)
            hour_of_day = t.hour
            day_of_week = t.weekday()
            rolling_1h = prev_occupied  # since logs are hourly
            exits_24h = sum((t - eh).total_seconds() <= 24*3600 for eh in exits_history)

            bed_rows.append({
                "Bed_id": bed,
                "patient_id": current_patient,
                "log_time": t.strftime("%Y-%m-%d %H:%M:%S"),
                "On_bed": int(occupied),
                "time_since_change": time_since_change,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "rolling_1h": rolling_1h,
                "exits_24h": exits_24h
            })

            prev_occupied = occupied

        # add future_occupancy target
        for i, row in enumerate(bed_rows):
            if i + horizon_steps < len(bed_rows):
                row["future_occupancy"] = bed_rows[i + horizon_steps]["On_bed"]
            else:
                row["future_occupancy"] = None  # can't know future at end
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_bed_logs(num_beds=13, days=7, freq_minutes=60, horizon_hours=4)
    df.to_csv("synthetic_bedlogs_features.csv", index=False)
    print("✅ Synthetic data with features + future_occupancy saved to synthetic_bedlogs_features.csv")
