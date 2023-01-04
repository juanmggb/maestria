import numpy as np

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