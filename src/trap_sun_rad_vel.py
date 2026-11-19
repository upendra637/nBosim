import numpy as np
import pandas as pd

# ============================================================
# 1. Read the TRAPPIST-1 CSV file
# ============================================================

file_path = "/Users/ankitpanda/Desktop/trappist_model_data/TRAPPIST1_simulation_new.csv"

df = pd.read_csv(file_path)

# ============================================================
# 2. Extract the TRAPPIST-1 star's velocity components (m/s)
# ============================================================

star_vx = df["TRAPPIST-1_vx_m_s"].to_numpy()
star_vy = df["TRAPPIST-1_vy_m_s"].to_numpy()
star_vz = df["TRAPPIST-1_vz_m_s"].to_numpy()

# ============================================================
# 3. Extract the COM velocity components (m/s)
# ============================================================

com_vx = df["COM_vx_m_s"].to_numpy()
com_vy = df["COM_vy_m_s"].to_numpy()
com_vz = df["COM_vz_m_s"].to_numpy()

# ============================================================
# 4. Calculate relative velocity components (m/s)
# ============================================================

relative_vx = star_vx - com_vx
relative_vy = star_vy - com_vy
relative_vz = star_vz - com_vz

# ============================================================
# 5. Calculate the magnitude of relative velocity (m/s)
# ============================================================

star_COM_velocity_m_s = np.sqrt(
    relative_vx**2
    + relative_vy**2
    + relative_vz**2
)

# Convert to km/s
star_COM_velocity_km_s = star_COM_velocity_m_s / 1000

# ============================================================
# 6. Add results to the DataFrame
# ============================================================

df["TRAPPIST-1_COM_relative_vx_m_s"] = relative_vx
df["TRAPPIST-1_COM_relative_vy_m_s"] = relative_vy
df["TRAPPIST-1_COM_relative_vz_m_s"] = relative_vz

df["TRAPPIST-1_COM_relative_velocity_m_s"] = star_COM_velocity_m_s
df["TRAPPIST-1_COM_relative_velocity_km_s"] = star_COM_velocity_km_s

# ============================================================
# 7. Display the first 10 results
# ============================================================

print("TRAPPIST-1 star velocity relative to COM")
print("------------------------------------------------")

print(
    df[[
        "Time_years",
        "TRAPPIST-1_COM_relative_vx_m_s",
        "TRAPPIST-1_COM_relative_vy_m_s",
        "TRAPPIST-1_COM_relative_vz_m_s",
        "TRAPPIST-1_COM_relative_velocity_m_s",
        "TRAPPIST-1_COM_relative_velocity_km_s"
    ]].head(10)
)

# ============================================================
# 8. Calculate velocity statistics
# ============================================================

print("\nRelative velocity statistics")
print("------------------------------------------------")

print(f"Minimum speed: {star_COM_velocity_m_s.min():.6f} m/s")
print(f"Maximum speed: {star_COM_velocity_m_s.max():.6f} m/s")
print(f"Average speed: {star_COM_velocity_m_s.mean():.6f} m/s")

print(f"\nMinimum speed: {star_COM_velocity_km_s.min():.8f} km/s")
print(f"Maximum speed: {star_COM_velocity_km_s.max():.8f} km/s")
print(f"Average speed: {star_COM_velocity_km_s.mean():.8f} km/s")

# ============================================================
# 9. Save the results to a new CSV file
# ============================================================

output_path = (
    "/Users/ankitpanda/Desktop/"
    "TRAPPIST1_COM_relative_velocity_new.csv"
)

df.to_csv(output_path, index=False)

print("\nRelative velocity data saved to:")
print(output_path)