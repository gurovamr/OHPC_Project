import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import math
import os
import sys
import csv
import time

data = pd.read_csv("data_OptiMinds.csv")  
print(data.head())

# Convert fractional years to datetime to understand the data better
def fractional_year_to_date(year):
    year_int = int(year)
    remainder = year - year_int
    start_of_year = datetime.datetime(year_int, 1, 1)
    days_in_year = (datetime.datetime(year_int + 1, 1, 1) - start_of_year).days
    return start_of_year + datetime.timedelta(days=remainder * days_in_year)

# Apply conversion
data['Date'] = data['Time'].apply(fractional_year_to_date)

# Drop the original 'Time' column if not needed
#data.drop(columns=['Time'], inplace=True)

print(data.head())

### Get the small subset to test functions locally
subset_random = data.sample(n=1000, random_state=42)
print(subset_random.head())

# Solar Cycle Model
def solar_cycle_model(t, params, num_cycles=10):
    """
    Models the solar cycles based on the given parameters.
    
    Args:
        t (ndarray): Array of time points.
        params (ndarray): Array of 3*num_cycles parameters 
                          [T0_1, Ts_1, Td_1, ..., T0_n, Ts_n, Td_n].
        num_cycles (int): Number of solar cycles (default is 10).
     
    Returns:
        ndarray: Predicted values of x(t).
    """
    predicted_values = np.zeros_like(t)

    for k in range(num_cycles):
        T0 = params[3 * k]       # Start time of cycle k
        Ts = params[3 * k + 1]   # Rising time of cycle k
        Td = params[3 * k + 2]   # Declining time of cycle k

        # Compute the contribution of cycle k
        x_k = ((t - T0) / Ts) ** 2 * np.exp(-((t - T0) / Td) ** 2)

        # Set contributions to zero outside the valid range
        x_k[(t < T0) | (t > T0 + Ts + Td)] = 0
        predicted_values += x_k