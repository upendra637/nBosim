import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# 1. Read the CSV file
# ============================================================

file_path = "/Users/ankitpanda/Desktop/solar_system_data/sun_jupiter_simulation_data.csv"

df = pd.read_csv(file_path)
time_years = df["Time_years"]
# ============================================================
# 2. Extract Sun's velocity components relative to COM (m/s)
# ============================================================

sun_vx = df["Sun_vx_m_s"]
sun_vy = df["Sun_vy_m_s"]
sun_vz = df["Sun_vz_m_s"]

# ============================================================
# 3. Calculate Sun's velocity magnitude at every time step
# ============================================================

sun_COM_velocity_m_s = np.sqrt(
    sun_vx**2 + sun_vy**2 + sun_vz**2
)

# Convert velocity from m/s to km/s
sun_COM_velocity_km_s = sun_COM_velocity_m_s / 1000

# ============================================================
# 4. Add the results to the DataFrame
# ============================================================

df["Sun_COM_velocity_m_s"] = sun_COM_velocity_m_s
df["Sun_COM_velocity_km_s"] = sun_COM_velocity_km_s

# ============================================================
# 5. Print the first 10 results
# ============================================================

print("Sun's velocity relative to COM at each time step")
print("------------------------------------------------")

print(
    df[[
        "Time_years",
        "Sun_COM_velocity_m_s",
        "Sun_COM_velocity_km_s"
    ]].head(10)
)

# ============================================================
# 6. Print minimum, maximum and average velocity
# ============================================================

print("\nVelocity statistics")
print("------------------------------------------------")

print(f"Minimum velocity: {sun_COM_velocity_m_s.min():.6f} m/s")
print(f"Maximum velocity: {sun_COM_velocity_m_s.max():.6f} m/s")
print(f"Average velocity: {sun_COM_velocity_m_s.mean():.6f} m/s")

print(f"\nMinimum velocity: {sun_COM_velocity_km_s.min():.6f} km/s")
print(f"Maximum velocity: {sun_COM_velocity_km_s.max():.6f} km/s")
print(f"Average velocity: {sun_COM_velocity_km_s.mean():.6f} km/s")

# ============================================================
# 4. Plot Sun-COM speed versus time
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    sun_COM_velocity_km_s,
    linewidth=1.5,
    label="Sun-COM speed"
)

plt.xlabel("Time (years)", fontsize=12)
plt.ylabel("Sun-COM speed (km/s)", fontsize=12)

plt.title("Sun's Speed Relative to COM vs Time", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()

# ============================================================
# 5. Save the plot as a PDF on the Desktop
# ============================================================

plt.savefig(
    "/Users/ankitpanda/Desktop/Sun_COM_velocity_vs_time.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
'''
# ============================================================
# 7. Save the updated CSV file
# ============================================================

output_path = "/Users/ankitpanda/Desktop/sun_jupiter_simulation_data.csv"

df.to_csv(output_path, index=False)

print("\nUpdated CSV saved successfully.")'''