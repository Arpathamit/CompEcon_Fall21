import numpy as np


##########################################################################################
#Household's Problem (4.6) (4.9) (4.10)

#(4.6)
def get_c(r, w, b_s, b_sp1, n_s):
    '''
    Find consumption using the budget constraint and the choice of savings (b_sp1)
    Eqn (4.6)
    '''
    c = (1 + r) * b_s + w * n_s - b_sp1
    return c

def mu_l_func(n_s, l_tilde, chi, b_radius, v):
    '''
    Marginal utility of labor supply
    '''
    mu_n = (chi 
            * (b_radius / l_tilde) 
            * (n_s / l_tilde) ** (v - 1) 
            * (1 - (n_s / l_tilde) ** v) ** ((1 - v) / v))

    return mu_n

#just U'(c)
def mu_c_func(c, sigma):
    '''
    Marginal utility of consumption
    '''
    mu_c = c ** -sigma
    return mu_c


#(4.10)
def hh_foc_c(w, b_s, b_sp1, n_s, beta, sigma, r):
    '''
    Define the household first order conditions of c
    Eqn (4.10)
    '''
    #beta, sigma, r = params
    c = get_c(r, w, b_s, b_sp1, n_s)

    mu_c = mu_c_func(c, sigma)
    euler_error_c = mu_c[:-1] - beta * (1 + r) * mu_c[1:]

    return euler_error_c


def hh_foc_l(w, b_s, b_sp1, n_s, l_tilde, chi, b_radius, v, beta, sigma, r):
    '''
    Define the household first order conditions of n
    Eqn (4.9)
    '''
    #beta, sigma, r = params
    c = get_c(r, w, b_s, b_sp1, n_s)
    
    mu_c = mu_c_func(c, sigma)
    mu_n = mu_l_func(n_s, l_tilde, chi, b_radius, v)

    euler_error_l = w * mu_c - mu_n

    return euler_error_l 


def FOCs(w, b_s, b_sp1, n_s, l_tilde, chi, b_radius, v, b_guess, beta, sigma, r):
    #beta, sigma, r = params
    b_s = np.append(b_guess, b_sp1)
    b_sp1 = np.append(b_sp1, 0)

    #args_params = (beta, sigma, r)
    b_errors = hh_foc_c(w, b_s, b_sp1, n_s, beta, sigma, r)
    n_errors = hh_foc_l(w, b_s, b_sp1, n_s, l_tilde, chi, b_radius, v, beta, sigma, r)
    euler_error = np.append(b_errors, n_errors)
    
    return euler_error

##########################################################################################


##########################################################################################
#Firms' problem: (4.13) and (4.14) at ss
def get_r(K, L, alpha, delta, A):
    '''
    Compute the interest rate from the firm's FOC
    '''
    #alpha, delta, A = params_firms

    r = alpha * A * (L / K) ** (1 - alpha) - delta
    return r


def get_w(r, alpha, delta, A):
    '''
    Solve for the w that is consistent with r from the firm's FOC
    '''
    #alpha, delta, A = params_firms
    w = (1 - alpha) * A * ((alpha * A) / (r + delta)) ** (alpha / (1 - alpha))
    return w
##########################################################################################


##########################################################################################
#Market clearing condition: (4.15), (4.16) at ss
def get_L(n):
    '''
    Function to compute aggregate labor supplied
    '''
    L = n.sum()
    return L


def get_K(b):
    '''
    Function to compute aggregate capital supplied
    '''
    K = b.sum()
    return K

#leave it here in case need it
def get_C(c):
    # Function to compute aggregate capital supplied
    C = c.sum()

    return C

def get_I(K, delta):
    '''
    Function to compute aggregate capital Saving
    '''
    I = K - ((1 - delta)*K)
    return I

#not sure
def get_Y(b_s, C, I):
    '''
    Function to compute aggregate capital Income
    '''
    Y = C + I
    return Y
##########################################################################################


