import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Read the CSV file
# ============================================================

file_path = (
    "/Users/ankitpanda/Desktop/"
    "trappist_model_data/TRAPPIST1_COM_relative_velocity_new.csv"
)

df = pd.read_csv(file_path)

# ============================================================
# 2. Extract time and relative velocity
# ============================================================

time_years = df["Time_years"].to_numpy()

velocity_m_s = df[
    "TRAPPIST-1_COM_relative_velocity_m_s"
].to_numpy()

velocity_km_s = df[
    "TRAPPIST-1_COM_relative_velocity_km_s"
].to_numpy()

# ============================================================
# 3. Plot velocity versus time in m/s
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(time_years, velocity_m_s, linewidth=1)

plt.xlabel("Time (Earth years)")
plt.ylabel("Velocity relative to COM (m/s)")
plt.title("TRAPPIST-1 Star Velocity Relative to COM vs Time")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "/Users/ankitpanda/Desktop/"
    "TRAPPIST1_COM_velocity_ms_new.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 4. Plot velocity versus time in km/s
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(time_years, velocity_km_s, linewidth=1)

plt.xlabel("Time (Earth years)")
plt.ylabel("Velocity relative to COM (km/s)")
plt.title("TRAPPIST-1 Star Velocity Relative to COM vs Time")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "/Users/ankitpanda/Desktop/"
    "TRAPPIST1_COM_velocity_kms_new.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()