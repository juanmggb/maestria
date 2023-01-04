import numpy as np

# Function to apply the solver method
def euler(f, x0, t0, tf, dt, u, *kargs):

    """
    Solve a system of differential equations using the Euler method.
    
    Parameters:
    f (function): The derivative function.
    x0 (ndarray): The initial state.
    t0 (float): The initial time.
    tf (float): The final time.
    dt (float): The time step.
    u (function): The input function.
    
    Returns:
    ndarray: The state at each time step.
    """
    t = np.arange(t0, tf, dt)
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
    for i in range(1, len(t)):
        # print(x[i,:])
        x[i, :] = x[i-1, :] + dt * f(x[i-1, :], t[i-1], u(t[i-1]), *kargs)
    return x
