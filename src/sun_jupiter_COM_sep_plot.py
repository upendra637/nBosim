import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# 1. Read the CSV file
# ============================================================

file_path = "/Users/ankitpanda/Desktop/sun_jupiter_simulation_data.csv"

df = pd.read_csv(file_path)

# ============================================================
# 2. Extract Sun's position relative to COM (in metres)
# ============================================================

sun_x = df["Sun_x_m"]
sun_y = df["Sun_y_m"]
sun_z = df["Sun_z_m"]

# ============================================================
# 3. Calculate Sun-COM separation at every time step
# ============================================================

sun_COM_separation_m = np.sqrt(
    sun_x**2 + sun_y**2 + sun_z**2
)

# Convert separation from metres to AU
AU = 1.495978707e11

sun_COM_separation_AU = sun_COM_separation_m / AU

# ============================================================
# 4. Add the results to the DataFrame
# ============================================================

df["Sun_COM_separation_m"] = sun_COM_separation_m
df["Sun_COM_separation_AU"] = sun_COM_separation_AU

# ============================================================
# 5. Print the first few results
# ============================================================

print("Sun-COM separation at each time step")
print("--------------------------------------------")

print(
    df[[
        "Time_years",
        "Sun_COM_separation_m",
        "Sun_COM_separation_AU"
    ]].head(10)
)

# ============================================================
# 6. Print minimum, maximum and average separation
# ============================================================

print("\nSeparation statistics")
print("--------------------------------------------")

print(f"Minimum separation: {sun_COM_separation_m.min():.6e} m")
print(f"Maximum separation: {sun_COM_separation_m.max():.6e} m")
print(f"Average separation: {sun_COM_separation_m.mean():.6e} m")

print(f"\nMinimum separation: {sun_COM_separation_AU.min():.8f} AU")
print(f"Maximum separation: {sun_COM_separation_AU.max():.8f} AU")
print(f"Average separation: {sun_COM_separation_AU.mean():.8f} AU")

'''# ============================================================
# 7. Save the updated CSV file
# ============================================================

output_path = "/Users/ankitpanda/Desktop/sun_jupiter_simulation_data.csv"

df.to_csv(output_path, index=False)

print("\nUpdated CSV saved successfully.")'''


###Plotting Sun-COM separation vs time
# ============================================================
# 7. Plot Sun-COM separation versus time
# ============================================================

time_years = df["Time_years"]

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    sun_COM_separation_AU,
    linewidth=1.5,
    label="Sun-COM separation"
)

plt.xlabel("Time (years)", fontsize=12)
plt.ylabel("Sun-COM separation (AU)", fontsize=12)

plt.title("Sun-COM Separation vs Time", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()

'''# Save the plot as a PDF on the Desktop
plt.savefig(
    "/Users/ankitpanda/Desktop/Sun_COM_separation_vs_time.pdf",
    dpi=300,
    bbox_inches="tight"
)'''

plt.show()