import pandas as pd
import numpy as np  
from scipy.signal import lombscargle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

data = pd.read_csv('period.csv')  # Load your data from a CSV file

time = data['Time_years'].values  # Assuming your time column is named 'time'
separation = data['Sun_COM_separation_AU'].values  # Assuming your separation column is named 'separation'

plt.plot(time, separation)
plt.xlabel('Time')
plt.ylabel('Sun-COM Separation (AU)')
plt.title('Sun-COM Separation Over Time')
plt.show()

# Perform Lomb-Scargle periodogram analysis

period = np.linspace(0.001, 170, 500)  # Period range in days
# # Compute the Lomb-Scargle periodogram
angular_frequencies = 2 * np.pi / period
power = lombscargle(time, separation, angular_frequencies)  # Compute the Lomb-Scargle periodogram


# Plot the Lomb-Scargle periodogram
plt.figure()
plt.plot(period, power)
plt.xlabel('Period (years)',fontsize=12)
plt.ylabel('Power',fontsize=12)
plt.title('Lomb-Scargle Periodogram',fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()
