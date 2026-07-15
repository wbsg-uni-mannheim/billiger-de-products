from codecarbon import OfflineEmissionsTracker
import time
import os
import json
import pandas as pd
import torch
def run_with_tracking(job_name, func, *args, gpt_usage=False, electricity_price_eur_per_kwh=0.30, **kwargs):

    os.makedirs("data/efficiency_tracker", exist_ok=True)
    json_path = f"data/efficiency_tracker/{job_name}.json"
    csv_path = f"data/efficiency_tracker/{job_name}.csv"

    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        output_file=csv_path
    )
    # ---- GPU MEMORY RESET (BEFORE TRAINING) ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    tracker.start()


    func(*args, **kwargs)  

    tracker.stop()
    end_time = time.time()
    runtime_sec = time.time() - start_time

    # ---- PEAK GPU MEMORY (AFTER TRAINING) ----
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        max_memory_mb = None

    # Calculate energy and costs
    emission_df = pd.read_csv(csv_path)
    energy_kwh = emission_df["energy_consumed"].iloc[-1]
    emissions_kg = emission_df["emissions"].iloc[-1]
    energy_cost_eur = energy_kwh * electricity_price_eur_per_kwh

    if not gpt_usage:
        total_cost_eur = energy_cost_eur
    else:
        cost = pd.read_csv("dataset_quality_test.csv")
        gpt_cost = cost["Costs"].sum()

    total_cost_eur = energy_cost_eur + gpt_cost

    # Log result
    record = {
        "job_name": job_name,
        "runtime_sec": round(runtime_sec, 3),
        "max_memory_mb": None if max_memory_mb is None else round(max_memory_mb, 3),
        "energy_kwh": round(energy_kwh, 6),
        "emissions_kg": round(emissions_kg, 6),
        "energy_cost_eur": round(energy_cost_eur, 4),
        "total_cost_eur": round(total_cost_eur, 4),
    }

    # Append or create JSON file
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(record)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    mem_str = "CPU" if max_memory_mb is None else f"{max_memory_mb:.2f} MB"
    print(f"Runtime: {runtime_sec:.2f}s | Max Memory: {mem_str} MB")
    print(f"Energy: {energy_kwh:.6f} kWh | CO₂: {emissions_kg:.6f} kg | Total Cost: {total_cost_eur:.4f} €")
    print(f"Results appended to: {json_path}")

if __name__ == "__main__":
    #TODO : Example usage
    pass