# Import modules
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Import functions
from utilis.solvers import euler
from utilis.models import inhibition_model_fb

# Generic functions #############################################################################

# Function to calculate final concentration
def final_conc(x, s, p=None):
    """
    Calculates the final concentrations of biomass, substrate and product.
     Args:
        x (numpy array): The biomass concentrations
        s (numpy array): The substrate concentrations
        p (numpy array): The product concentrations
        
    Returns:
        (tuple): A tuple with the final concentrations of biomass, substrate and product
    """

    try:
        return (x[-1],s[-1], p[-1])
    except TypeError:
        return (x[-1],s[-1])

# Function to calculate the MSE
def MSE(data, predicted):
    """
    Calculate the MSE between a data set and model predictions.
    Args: 
        data (pandas data frame): a data set with column: x, s and p
        predictions (pandas data frame): a data set with columns: x, s and p
    Returns: 
        (float): The MSE
    """
    data = np.array(data)
    predicted = np.array(predicted)
    return np.sum((data - predicted)**2)/len(data)

# Bath functions #################################################################################
# Function to create data frame from numpy arrays
def create_df(time, x, s, p=None):

    """
    Creates a pandas data frame from numpy arrays.
    Args: 
        time (numpy array): Time values
        x (numpy array): The biomass concentrations
        s (numpy array): The substrate concentrations
        p (numpy array): The product concentrations
    """

    if p is not None:
        predicted = {'t': [], 'x':[], 's':[], 'p':[]}
    else: 
        predicted = {'t': [], 'x':[], 's':[]}

    for i in range(len(time)):

        predicted['t'].append(time[i])
        predicted['x'].append(x[i])
        predicted['s'].append(s[i])
        if p is not None:
            predicted['p'].append(p[i])

    predicted = pd.DataFrame(predicted)
    return predicted

# Function to plot predictions 
def plot_data(data):
    """
    Plot data from a batch simulation.
    Args: 
        data (pandas data frame): A data frame with columns: t, x, s and p
    Returns: 
        None
    """
    plt.figure(figsize=(10,5))
    plt.plot(data.t, data.x, label='x')
    plt.plot(data.t, data.s, label='s')
    try:
        plt.plot(data.t, data.p, label='p')
    except AttributeError:
        print('No product "p" in data')
    plt.ylabel('Concentration (g/L)')
    plt.xlabel('Time (h)')
    plt.grid()
    plt.legend()
    plt.show()

# Function to add noise to data 
def add_noise(predicted, x_noise, s_noise, p_noise=None):

    """
    Add normal noise to data frame predicted.
    Args: 
        predicted (pandas data frame): A data frame with columns: t, x, s and p
        x_noise (float): Noise for columns x
        s_noise (float): Noise for columns s
        p_noise (float): Noise for columns p
    Returns: 
        predicted_noise (pandas data frame): A copy of data frame predicted with added noise
    """

    predicted_noise = predicted.copy() 

    predicted_noise.x = predicted_noise.x + np.random.normal(loc=0, scale=x_noise, size=len(predicted_noise.x))
    predicted_noise.s = predicted_noise.s + np.random.normal(loc=0, scale=s_noise, size=len(predicted_noise.s))
    try:
        predicted_noise.p = predicted_noise.p + np.random.normal(loc=0, scale=p_noise, size=len(predicted_noise.p))
    except AttributeError:
        print('No product "p" in data')

    return predicted_noise



# Fed bath functions #################################################################################
#  Function to create data frame from numpy arrays
def create_df_fb(time, x, s, V=None):

    """
    Creates a pandas data frame from numpy arrays.
    Args: 
        time (numpy array): Time values
        x (numpy array): The biomass concentrations
        s (numpy array): The substrate concentrations
        p (numpy array): The product concentrations
    """

    if V is not None:
        predicted = {'t': [], 'x':[], 's':[], 'V':[]}
    else: 
        predicted = {'t': [], 'x':[], 's':[]}

    for i in range(len(time)):

        predicted['t'].append(time[i])
        predicted['x'].append(x[i])
        predicted['s'].append(s[i])
        if V is not None:
            predicted['V'].append(V[i])

    predicted = pd.DataFrame(predicted)
    return predicted

# Function to plot predictions 
def plot_data_fb(data):
    
    """
    Plot data from a fed batch simulation.
    Args: 
        data (pandas data frame): A data frame with columns: t, x and s 
    Returns: 
        None
    """

    plt.figure(figsize=(10,5))
    plt.plot(data.t, data.x, label='x')
    plt.plot(data.t, data.s, label='s')
    plt.ylabel('Concentration (g/L)')
    plt.xlabel('Time (h)')
    plt.grid()
    plt.legend()
    plt.show()

def fitness_function_fb(u, x0, ti, tf, dt, mu_max, yxs, ks, ypx, ki, sf):
    """
    Calculate the biomass production at the end of the fed batch fermentation.
    Args:
        u (int): The input value to the bioreactor 
        x0 (numpy array): Initial conditions to the euler method. Values of x and s
        ti (float): Initial time of simulation
        tf (float): Final time of simulation
        dt (float): Delta time for the euler method
        mu_max (float): Kinetic parameter
        yxs (float): Kinetic parameter
        ks (float): Kinetic parameter
        ki (float): Kinetic parameter
        sf (float): Kinetic parameter
    """
    x, _, V = euler(inhibition_model_fb, x0, ti, tf, dt, lambda t: u, mu_max, yxs, ks, ypx, ki, sf).T
    return x[-1]*V[-1]