# ============================================================
# SOLAR ECLIPSE ANALYSIS FROM CHENNAI (2027-2030)
# ============================================================

# 1. Import required libraries

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from skyfield.api import load, wgs84


# 2. Chennai location

latitude = 13.0827
longitude = 80.2707

IST = ZoneInfo("Asia/Kolkata")


# 3. Solar eclipse dates (2027-2030)

dates = [
    "2027-02-06",
    "2027-08-02",
    "2028-01-26",
    "2028-07-22",
    "2029-01-14",
    "2029-06-12",
    "2029-07-11",
    "2029-12-05",
    "2030-06-01",
    "2030-11-25"
]


# 4. Global eclipse types (NASA reference)

global_types = {
    "2027-02-06": "Annular",
    "2027-08-02": "Total",
    "2028-01-26": "Annular",
    "2028-07-22": "Total",
    "2029-01-14": "Partial",
    "2029-06-12": "Partial",
    "2029-07-11": "Partial",
    "2029-12-05": "Partial",
    "2030-06-01": "Annular",
    "2030-11-25": "Total"
}


# 5. Chennai reference data

# None means no eclipse is expected to be visible.

chennai_reference = {
    "2027-02-06": None,
    "2027-08-02": "Partial",
    "2028-01-26": None,
    "2028-07-22": "Partial",
    "2029-01-14": None,
    "2029-06-12": None,
    "2029-07-11": None,
    "2029-12-05": None,
    "2030-06-01": None,
    "2030-11-25": None
}


# 6. Load astronomical data

print("Loading JPL ephemeris...")

eph = load("de421.bsp")
ts = load.timescale()

earth = eph["earth"]
sun = eph["sun"]
moon = eph["moon"]

# Define Chennai as the observing location.

chennai = earth + wgs84.latlon(latitude, longitude)


# 7. Convert physical radii from km to AU

AU_KM = 149597870.7

sun_radius = 695700 / AU_KM
moon_radius = 1737.4 / AU_KM


# 8. Determine the type of solar eclipse

def classify_eclipse(separation, sun_angle, moon_angle):

    # No overlap between the Sun and Moon.

    if separation >= sun_angle + moon_angle:
        return "None"

    # One disk completely covers or lies inside the other.

    if separation <= abs(sun_angle - moon_angle):

        if moon_angle >= sun_angle:
            return "Total"

        else:
            return "Annular"

    # Partial overlap.

    return "Partial"


# 9. Analyse one eclipse date

def analyse_eclipse(date_text):

    # Convert date into a Python date object.

    date = datetime.strptime(date_text, "%Y-%m-%d").date()

    # Start of the day in Chennai local time.

    start = datetime(
        date.year, date.month, date.day,
        tzinfo=IST
    )

    # Generate observations at one-minute intervals.

    seconds = np.arange(0, 86400, 60)

    observation_times = [
        start + timedelta(seconds=int(s))
        for s in seconds
    ]

    # Convert observation times into Skyfield time objects.

    times = ts.from_datetimes(observation_times)

    # Calculate apparent positions of the Sun and Moon.

    sun_position = chennai.at(times).observe(sun).apparent()
    moon_position = chennai.at(times).observe(moon).apparent()

    # Angular separation between the centres of the disks.

    separation = sun_position.separation_from(
        moon_position
    ).radians

    # Calculate apparent angular radii.

    sun_angle = np.arcsin(
        sun_radius / sun_position.distance().au
    )

    moon_angle = np.arcsin(
        moon_radius / moon_position.distance().au
    )

    # Difference between disk separation and combined radii.
    # Negative value means that the disks overlap.

    gap = separation - sun_angle - moon_angle

    # Calculate the Sun's altitude above the horizon.

    sun_altitude = sun_position.altaz()[0].degrees

    # Check whether the disks overlap.

    overlap = gap < 0

    # Check whether the Sun is above the horizon.

    sun_visible = sun_altitude > 0

    # An eclipse is observable only when both conditions hold.

    visible = overlap & sun_visible

    # Case 1: No overlap throughout the day.

    if not np.any(overlap):

        return "NOT VISIBLE", "None"

    # Case 2: Overlap occurs, but only below the horizon.

    if not np.any(visible):

        return "NOT VISIBLE", "None"

    # Case 3: Find the visible observation with the
    # smallest angular separation between the Sun and Moon.

    visible_indices = np.where(visible)[0]

    best_index = visible_indices[
        np.argmin(separation[visible_indices])
    ]

    # Classify the eclipse at this observation.

    eclipse_type = classify_eclipse(
        separation[best_index],
        sun_angle[best_index],
        moon_angle[best_index]
    )

    if eclipse_type == "None":

        return "NOT VISIBLE", "None"

    return "VISIBLE", eclipse_type


# 10. Compare predictions with reference data

print("\nSOLAR ECLIPSE ANALYSIS: CHENNAI")
print("=" * 95)

print(
    f"{'Date':<15}"
    f"{'Global Type':<15}"
    f"{'Predicted':<17}"
    f"{'Predicted Type':<18}"
    f"{'Reference Type':<18}"
    f"{'Validation'}"
)

print("-" * 95)


for date in dates:

    # Calculate Chennai visibility and eclipse type.

    predicted_status, predicted_type = analyse_eclipse(date)

    # Get the reference eclipse type.

    reference_type = chennai_reference[date]

    if reference_type is None:
        reference_status = "NOT VISIBLE"
        reference_type = "None"

    else:
        reference_status = "VISIBLE"

    # Compare predicted and reference results.

    if (
        predicted_status == reference_status
        and predicted_type == reference_type
    ):
        validation = "MATCH"

    else:
        validation = "MISMATCH"

    # Display results.

    print(
        f"{date:<15}"
        f"{global_types[date]:<15}"
        f"{predicted_status:<17}"
        f"{predicted_type:<18}"
        f"{reference_type:<18}"
        f"{validation}"
    )


# 11. Reference sources

print("\n\nREFERENCE SOURCES")

print(
    "NASA Global Eclipse Catalogue:\n"
    "https://eclipse.gsfc.nasa.gov/SEdecade/SEdecade2021.html"
)

print(
    "\nChennai Eclipse References:\n"
    "https://theskylive.com/solar-eclipse"
    "?geoid=1264527&id=2027-08-02"
)

print(
    "https://www.timeanddate.com/eclipse/in/india/"
    "chennai?iso=20280722"
)

print(
    "https://nexttotality.com/en/cities/chennai/"
)