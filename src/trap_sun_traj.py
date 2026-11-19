import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Read the CSV file
# ============================================================

file_path = "/Users/ankitpanda/Desktop/trappist_model_data/TRAPPIST1_simulation_new.csv"

df = pd.read_csv(file_path)

AU = 1.495978707e11  # metres

# ============================================================
# 2. Extract star and COM positions
# ============================================================

star_x = df["TRAPPIST-1_x_m"].to_numpy()
star_y = df["TRAPPIST-1_y_m"].to_numpy()
star_z = df["TRAPPIST-1_z_m"].to_numpy()

com_x = df["COM_x_m"].to_numpy()
com_y = df["COM_y_m"].to_numpy()
com_z = df["COM_z_m"].to_numpy()

# Star position relative to COM, converted to AU
star_x_com = (star_x - com_x) / AU
star_y_com = (star_y - com_y) / AU
star_z_com = (star_z - com_z) / AU

# ============================================================
# 3. Extract time
# ============================================================

time_years = df["Time_years"].to_numpy()

# ============================================================
# 4. Plot trajectory in the XY plane
# ============================================================

plt.figure(figsize=(9, 7))

plt.plot(star_x_com, star_z_com, label="TRAPPIST-1 trajectory")

plt.scatter(
    star_x_com[0], star_z_com[0],
    color="green", label="Initial position"
)

plt.scatter(
    star_x_com[-1], star_z_com[-1],
    color="red", label="Final position"
)

plt.scatter(
    0, 0, color="black", marker="x",
    s=80, label="Centre of Mass"
)

plt.xlabel("X position relative to COM (AU)")
plt.ylabel("Z position relative to COM (AU)")
plt.title("TRAPPIST-1 Star Trajectory Around the COM")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "/Users/ankitpanda/Desktop/TRAPPIST1_Sun_COM_trajectory.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 5. Plot the 3D trajectory
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    star_x_com, star_y_com, star_z_com,
    label="TRAPPIST-1 trajectory"
)

ax.scatter(
    0, 0, 0, color="black", marker="x",
    s=80, label="Centre of Mass"
)

ax.set_xlabel("X position relative to COM (AU)")
ax.set_ylabel("Y position relative to COM (AU)")
ax.set_zlabel("Z position relative to COM (AU)")
ax.set_title("3D TRAPPIST-1 Trajectory Around the COM")

ax.legend()
plt.tight_layout()

plt.savefig(
    "/Users/ankitpanda/Desktop/TRAPPIST1_Sun_COM_trajectory_3D.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




###########################################################################
###########################################################################
###########################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Read the TRAPPIST-1 simulation CSV file
# ============================================================

file_path = (
    "/Users/ankitpanda/Desktop/"
    "trappist_model_data/TRAPPIST1_simulation_new.csv"
)

df = pd.read_csv(file_path)

AU = 1.495978707e11  # Astronomical unit in metres

# ============================================================
# 2. Extract star position (metres)
# ============================================================

star_x = df["TRAPPIST-1_x_m"].to_numpy()
star_y = df["TRAPPIST-1_y_m"].to_numpy()
star_z = df["TRAPPIST-1_z_m"].to_numpy()

# ============================================================
# 3. Extract COM position (metres)
# ============================================================

com_x = df["COM_x_m"].to_numpy()
com_y = df["COM_y_m"].to_numpy()
com_z = df["COM_z_m"].to_numpy()

# ============================================================
# 4. Calculate star-COM separation at every time step
# ============================================================

dx = star_x - com_x
dy = star_y - com_y
dz = star_z - com_z

separation_m = np.sqrt(
    dx**2 + dy**2 + dz**2
)

# Convert separation to AU
separation_AU = separation_m / AU

# ============================================================
# 5. Create a separate DataFrame
# ============================================================

output_df = pd.DataFrame({
    "Time_seconds": df["Time_seconds"],
    "Time_years": df["Time_years"],
    "Star_COM_separation_m": separation_m,
    "Star_COM_separation_AU": separation_AU
})

'''# ============================================================
# 6. Save the separation data to a new CSV file
# ============================================================

output_path = (
    "/Users/ankitpanda/Desktop/"
    "TRAPPIST1_star_COM_separation_new.csv"
)

output_df.to_csv(output_path, index=False)

print("Separation data saved to:")
print(output_path)

print("\nFirst 10 rows:")
print(output_df.head(10).to_string(index=False))'''

print("\nSeparation statistics:")
print(f"Minimum separation: {separation_m.min():.6e} m")
print(f"Maximum separation: {separation_m.max():.6e} m")
print(f"Average separation: {separation_m.mean():.6e} m")

print(f"\nMinimum separation: {separation_AU.min():.8e} AU")
print(f"Maximum separation: {separation_AU.max():.8e} AU")
print(f"Average separation: {separation_AU.mean():.8e} AU")


# ============================================================
# 8. Plot separation versus time in AU
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(time_years, separation_AU, linewidth=1)

plt.xlabel("Time (Earth years)")
plt.ylabel("Star-COM separation (AU)")
plt.title("TRAPPIST-1 Star Separation from COM vs Time")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "/Users/ankitpanda/Desktop/"
    "TRAPPIST1_star_COM_separation_new.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()