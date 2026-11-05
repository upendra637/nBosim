# ============================================================
# LUNAR ECLIPSE ANALYSIS FROM CHENNAI (2027-2030)
# ============================================================

# 1. Import libraries

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
from skyfield.api import load, wgs84


# 2. Chennai location

latitude = 13.0827
longitude = 80.2707

IST = ZoneInfo("Asia/Kolkata")


# 3. Lunar eclipse dates from NASA (UTC dates)

dates = [
    "2027-02-20",
    "2027-07-18",
    "2027-08-17",
    "2028-01-12",
    "2028-07-06",
    "2028-12-31",
    "2029-06-26",
    "2029-12-20",
    "2030-06-15",
    "2030-12-09"
]


# 4. Global lunar eclipse types (NASA)

global_types = {
    "2027-02-20": "Penumbral",
    "2027-07-18": "Penumbral",
    "2027-08-17": "Penumbral",
    "2028-01-12": "Partial",
    "2028-07-06": "Partial",
    "2028-12-31": "Total",
    "2029-06-26": "Total",
    "2029-12-20": "Total",
    "2030-06-15": "Partial",
    "2030-12-09": "Penumbral"
}


# 5. Load astronomical data

print("Loading JPL ephemeris...")

eph = load("de421.bsp")
ts = load.timescale()

earth = eph["earth"]
sun = eph["sun"]
moon = eph["moon"]

# Observer located in Chennai

chennai = earth + wgs84.latlon(latitude, longitude)


# 6. Physical constants

AU_KM = 149597870.7

earth_radius = 6378.137
sun_radius = 695700.0
moon_radius = 1737.4


# 7. Classify lunar eclipse

def classify_eclipse(separation, umbra, penumbra, moon_angle):

    # Total eclipse:
    # Entire Moon lies inside Earth's umbra.

    if separation + moon_angle <= umbra:
        return "Total"

    # Partial eclipse:
    # Part of the Moon enters the umbra.

    if separation < umbra + moon_angle:
        return "Partial"

    # Penumbral eclipse:
    # Moon overlaps only the outer shadow.

    if separation < penumbra + moon_angle:
        return "Penumbral"

    return "None"


# 8. Analyse one lunar eclipse

def analyse_eclipse(date_text):

    # NASA dates refer to UTC.
    # Search 18 hours before and after the listed date
    # to include eclipses crossing midnight.

    date = datetime.strptime(
        date_text, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    start = date - timedelta(hours=18)

    # Generate observations every minute over 60 hours.

    minutes = np.arange(0, 60 * 60, 1)

    observation_times = [
        start + timedelta(minutes=int(m))
        for m in minutes
    ]

    times = ts.from_datetimes(observation_times)

    # Geocentric positions of the Sun and Moon

    sun_geo = earth.at(times).observe(sun).apparent()
    moon_geo = earth.at(times).observe(moon).apparent()

    # Earth's shadow points opposite to the Sun.
    # Angular separation between Moon and shadow centre.

    sun_direction = sun_geo.position.au
    moon_direction = moon_geo.position.au

    shadow_direction = -sun_direction

    dot_product = np.sum(
        shadow_direction * moon_direction, axis=0
    )

    shadow_length = np.linalg.norm(
        shadow_direction, axis=0
    )

    moon_length = np.linalg.norm(
        moon_direction, axis=0
    )

    cosine = dot_product / (shadow_length * moon_length)

    separation = np.arccos(
        np.clip(cosine, -1, 1)
    )

    # Distances from Earth to Sun and Moon (km)

    sun_distance = sun_geo.distance().au * AU_KM
    moon_distance = moon_geo.distance().au * AU_KM

    # Apparent angular radius of the Moon

    moon_angle = np.arcsin(
        moon_radius / moon_distance
    )

    # Horizontal parallax of Moon and Sun

    moon_parallax = np.arcsin(
        earth_radius / moon_distance
    )

    sun_parallax = np.arcsin(
        earth_radius / sun_distance
    )

    # Apparent angular radius of the Sun

    sun_angle = np.arcsin(
        sun_radius / sun_distance
    )

    # Earth's shadow radii at the Moon.
    # NASA conventional shadow enlargement: 2%

    penumbra = 1.02 * (
        0.998340 * moon_parallax
        + sun_angle
        + sun_parallax
    )

    umbra = 1.02 * (
        0.998340 * moon_parallax
        - sun_angle
        + sun_parallax
    )

    # Calculate Moon altitude from Chennai

    moon_local = chennai.at(times).observe(moon).apparent()

    moon_altitude = moon_local.altaz()[0].degrees

    # Check whether Moon overlaps Earth's penumbra

    overlap = separation < penumbra + moon_angle

    # Check whether Moon is above Chennai's horizon

    moon_visible = moon_altitude > 0

    # Both conditions must be satisfied

    visible = overlap & moon_visible

    if not np.any(overlap):
        return "NOT VISIBLE", "None"

    if not np.any(visible):
        return "NOT VISIBLE", "None"

    # Find visible observation with minimum separation

    visible_indices = np.where(visible)[0]

    best_index = visible_indices[
        np.argmin(separation[visible_indices])
    ]

    # Classify the locally visible eclipse

    eclipse_type = classify_eclipse(
        separation[best_index],
        umbra[best_index],
        penumbra[best_index],
        moon_angle[best_index]
    )

    return "VISIBLE", eclipse_type


# 9. Run the simulation

print("\nLUNAR ECLIPSE ANALYSIS: CHENNAI")
print("=" * 85)

print(
    f"{'Date (UTC)':<15}"
    f"{'Global Type':<15}"
    f"{'Chennai Visibility':<22}"
    f"{'Local Type':<15}"
)

print("-" * 85)

for date in dates:

    status, eclipse_type = analyse_eclipse(date)

    print(
        f"{date:<15}"
        f"{global_types[date]:<15}"
        f"{status:<22}"
        f"{eclipse_type:<15}"
    )


# 10. Reference sources

print("\nREFERENCE SOURCES")

print(
    "NASA Lunar Eclipse Catalogue:\n"
    "https://eclipse.gsfc.nasa.gov/LEdecade/"
    "LEdecade2021.html"
)

print(
    "\nChennai Eclipse Calendar:\n"
    "https://www.timeanddate.com/eclipse/in/india/chennai"
)
