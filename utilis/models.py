import numpy as np
import torch.nn as nn

# Monod model
def monod_model(y, t, u,  mu_max, yxs, ks, ypx):
    x, s, p = y 

    growth_rate = mu_max*(s/(ks + s))

    dx = growth_rate*x 
    ds = -1/yxs*growth_rate*x
    dp = ypx*growth_rate*x

    return np.array([dx, ds, dp])


# Inhibition model
def inhibition_model(y, t, u, mu_max, yxs, ks, ypx, ki):
    x, s, p = y 

    growth_rate = mu_max*(s/(ks + s + ki*s**2))

    dx = growth_rate*x 
    ds = -1/yxs*growth_rate*x
    dp = ypx*growth_rate*x

    return np.array([dx, ds, dp])


# Inhibition model for fed bach bioreactor
def inhibition_model_fb(y, t, u, mu_max, yxs, ks, ypx, ki, sf):
                         # mu_max, yxs, ks, ypx, ki, sf
    F = u

    x, s, V = y

    # This is the part that will be sustituted by an ANN likely
    growth_rate = mu_max*(s/(ks+s+ki*s**2))

    # Mass balances 
    dx = growth_rate*x - F*x/V
    ds = F/V*(sf-s) -1/yxs*growth_rate*x
    dV = F
    # dp = ypx*growth_rate*x 

    return np.array([dx, ds, dV])

# ANN for batch process



def fnn_model(input_dim, output_dim, hidden_dim, hidden_layers):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim)) 
        layers.append(nn.Softplus()) 
    layers.append(nn.Linear(hidden_dim, output_dim))
    net = nn.Sequential(*layers) 
    net.to('cpu').double()

    for m in net.modules():
        if type(m) == nn.Linear:
            # Initialize the weights of the Linear module using xavier_uniform_
            nn.init.xavier_uniform_(m.weight)
    return net



#@title bioreactor model
def inhibition_model_nn(t, x):
    b = x[:, 0]
    s = x[:, 1]
    p = x[:, 2]

    # Kinetic parameters
    mu = 1.2 # 1/h
    ks = 280 # g/L
    Yxs = 0.2 
    Ypx = 4 
    ki = 0.3

    # Mass balances
    db = mu*(s / (ks + s + ki*s**2))*b 
    ds = -1/Yxs*mu*(s / (ks + s + ki*s**2))*b 
    dp = Ypx*mu*(s / (ks + s + ki*s**2))*b

    return torch.stack((db, ds, dp), dim=-1)