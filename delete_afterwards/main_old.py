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
# Convert fractional years to datetime to understand the data better
def fractional_year_to_date(year):
    year_int = int(year)
    remainder = year - year_int
    start_of_year = datetime.datetime(year_int, 1, 1)
    days_in_year = (datetime.datetime(year_int + 1, 1, 1) - start_of_year).days
    return start_of_year + datetime.timedelta(days=remainder * days_in_year)

# Apply conversion
data['Date'] = data['Time'].apply(fractional_year_to_date)

subset_random = data.sample(n=1000, random_state=42)
print(subset_random.head())


def solar_cycle_model(t, params, num_cycles=10):
    """
    Models the solar cycles based on the given parameters, ensuring distinct parameters for each cycle.
    
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

        # Define the range of this cycle
        if k < num_cycles - 1:
            next_T0 = params[3 * (k + 1)]  # Start time of the next cycle
            cycle_mask = (t >= T0) & (t < next_T0)  # Prevent overlap with the next cycle
        else:
            cycle_mask = (t >= T0)  # Last cycle includes all times after T0

        t_cycle = t[cycle_mask]

        # Compute the contribution of cycle k
        if len(t_cycle) > 0:
            x_k = ((t_cycle - T0) / Ts) ** 2 * np.exp(-((t_cycle - T0) / Td) ** 2)
            predicted_values[cycle_mask] += x_k

    return predicted_values


# Loss Function: Mean Squared Error
def mse(params, t, observed_values):
    """
    Computes the Mean Squared Error (MSE) between observed data and model predictions.
    
    Args:
        params (ndarray): Model parameters [T0_1, Ts_1, Td_1, ..., T0_n, Ts_n, Td_n].
        t (ndarray): Time points of the observations.
        observed_values (ndarray): Observed sunspot data.
    
    Returns:
        float: The MSE value.
    """
    predicted_values = solar_cycle_model(t, params)
    return np.mean((observed_values - predicted_values) ** 2)

    def simulated_annealing(x0, T0, sigma, f, n_iter=10000, burn_in=5000):
    """
    Performs Simulated Annealing to optimize the solar cycle parameters.
    
    Args:
        x0 (ndarray): Initial parameter guess.
        T0 (float): Initial temperature.
        sigma (float): Proposal standard deviation.
        f (function): Loss function to minimize.
        n_iter (int): Total number of iterations.
        burn_in (int): Burn-in period.
    
    Returns:
        tuple: Optimized parameters and loss function evolution.
    """
    x = x0.copy()
    n_params = len(x0)
    temperature = T0
    losses = []
    best_x = x.copy()
    best_loss = f(x)

    for i in range(n_iter):
        # Propose new parameters
        x_new = x + np.random.normal(0, sigma, size=n_params)
        
        # Calculate the loss difference
        loss_old = f(x)
        loss_new = f(x_new)
        delta_e = loss_new - loss_old
        
        # Metropolis criterion
        if np.exp(-delta_e / temperature) >= np.random.rand():
            x = x_new
            if loss_new < best_loss:
                best_loss = loss_new
                best_x = x_new.copy()
        
        # Cool the temperature
        temperature = T0 / (1 + i / n_iter)
        losses.append(loss_old)
    
    # Discard burn-in period
    return best_x, np.array(losses[burn_in:])


    # Initial parameters from the paper
T0_initial = np.array([1755.2, 1766.06, 1775.5, 1784.9, 1798.4, 
                       1810.6, 1823.05, 1833.11, 1843.5, 1855.12])
Ts_initial = np.full(len(T0_initial), 0.3)
Td_initial = np.full(len(T0_initial), 5.0)
x0 = np.empty(len(T0_initial) * 3)
x0[::3] = T0_initial
x0[1::3] = Ts_initial
x0[2::3] = Td_initial

print("Initial Parameters:", x0)

# Extract and sort time points from subset_random
subset_random = subset_random.sort_values(by="Time")
t = subset_random['Time'].values
observed_values = subset_random['SN'].values


# Compute predictions using the model for all cycles
num_cycles = len(T0_initial)  # Number of cycles based on initial parameters
predicted_values = solar_cycle_model(t, x0, num_cycles=num_cycles)


# Plot observed vs. predicted values (all cycles)
plt.figure(figsize=(12, 6))
plt.scatter(t, observed_values, label="Observed Data", color="blue", alpha=0.7)
plt.plot(t, predicted_values, label="Solar Cycle Model (All Cycles)", color="orange", linestyle="--")
plt.title("Observed vs Solar Cycle Model")
plt.xlabel("Time")
plt.ylabel("Sunspot Values")
plt.legend()
plt.show()