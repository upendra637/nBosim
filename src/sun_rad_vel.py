
import numpy as np
import pandas as pd

# ============================================================
# 1. Read the original multi-planetary Solar System CSV file
# ============================================================

file_path = "/Users/ankitpanda/Desktop/solar_system_data/solar_system_data_real165yr.csv"


df = pd.read_csv(file_path)

'''# ============================================================
# 2. Extract the Sun's velocity components (m/s)
# ============================================================

sun_vx = df["Sun_vx_m_s"]
sun_vy = df["Sun_vy_m_s"]
sun_vz = df["Sun_vz_m_s"]

# ============================================================
# 3. Extract the COM velocity components (m/s)
# ============================================================

com_vx = df["COM_vx_m_s"]
com_vy = df["COM_vy_m_s"]
com_vz = df["COM_vz_m_s"]

# ============================================================
# 4. Calculate relative velocity components (m/s)
# ============================================================

relative_vx = sun_vx - com_vx
relative_vy = sun_vy - com_vy
relative_vz = sun_vz - com_vz

# ============================================================
# 5. Calculate the magnitude of relative velocity (m/s)
# ============================================================

sun_COM_velocity_m_s = np.sqrt(
    relative_vx**2
    + relative_vy**2
    + relative_vz**2
)

# Convert to km/s
sun_COM_velocity_km_s = sun_COM_velocity_m_s / 1000

# ============================================================
# 6. Add the results to the DataFrame
# ============================================================

df["Sun_COM_relative_vx_m_s"] = relative_vx
df["Sun_COM_relative_vy_m_s"] = relative_vy
df["Sun_COM_relative_vz_m_s"] = relative_vz

df["Sun_COM_relative_velocity_m_s"] = sun_COM_velocity_m_s
df["Sun_COM_relative_velocity_km_s"] = sun_COM_velocity_km_s

# ============================================================
# 7. Display the first 10 results
# ============================================================

print("Sun's velocity relative to COM")
print("------------------------------------------------")

print(
    df[[
        "Time_years",
        "Sun_COM_relative_vx_m_s",
        "Sun_COM_relative_vy_m_s",
        "Sun_COM_relative_vz_m_s",
        "Sun_COM_relative_velocity_m_s",
        "Sun_COM_relative_velocity_km_s"
    ]].head(10)
)

# ============================================================
# 8. Calculate velocity statistics
# ============================================================

print("\nRelative velocity statistics")
print("------------------------------------------------")

print(f"Minimum speed: {sun_COM_velocity_m_s.min():.6f} m/s")
print(f"Maximum speed: {sun_COM_velocity_m_s.max():.6f} m/s")
print(f"Average speed: {sun_COM_velocity_m_s.mean():.6f} m/s")

print(f"\nMinimum speed: {sun_COM_velocity_km_s.min():.8f} km/s")
print(f"Maximum speed: {sun_COM_velocity_km_s.max():.8f} km/s")
print(f"Average speed: {sun_COM_velocity_km_s.mean():.8f} km/s")

# ============================================================
# 9. Save the results to a new CSV file
# ============================================================

output_path = (
    "/Users/ankitpanda/Desktop/"
    "sun_COM_radial_velocity_1.csv"
)

df.to_csv(output_path, index=False)

print("\nRelative velocity data saved to:")
print(output_path)'''
############################################################
############################################################

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Read the saved CSV file
# ============================================================

file_path = "/Users/ankitpanda/Desktop/solar_system_data/sun_COM_radial_velocity_1.csv"

df = pd.read_csv(file_path)

# ============================================================
# 2. Extract time and Sun-COM relative speed
# ============================================================

time_years = df["Time_years"]

sun_COM_velocity_m_s = df["Sun_COM_relative_velocity_m_s"]

# ============================================================
# 3. Plot Sun-COM relative speed versus time
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time_years,
    sun_COM_velocity_m_s,
    linewidth=1.2,
    label="Sun's velocity relative to COM"
)

plt.xlabel("Time (years)", fontsize=12)
plt.ylabel("Relative speed (m/s)", fontsize=12)

plt.title(
    "Sun's Velocity Relative to the Solar System COM vs Time",
    fontsize=14
)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()

# ============================================================
# 4. Save the plot as a PDF
# ============================================================

plt.savefig(
    "/Users/ankitpanda/Desktop/Sun_COM_relative_velocity_vs_time.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()