#import os
#os.chdir(r'C:\Users\haimiti.aerfate\Desktop\ProblemSet9')
import necessary_equations as ne
import scipy.optimize as opt
import numpy as np

def SS_solver(r_guess, params):
    '''
    Solves for the SS
    '''
    l_tilde, chi, b_radius, v, b_guess, beta, sigma, S, A, alpha, delta = params
    xi = 0.8
    tol = 1e-8
    max_iter = 500
    dist = 7.0
    iterr = 0
    r = r_guess
    n_s_guess = np.ones(S)
    b_sp1_guess = np.ones(S)
    
    while (dist > tol) & (iterr < max_iter):
        w = ne.get_w(r, alpha, delta, A)
        

        foc_args = (l_tilde, chi, b_radius, v, b_guess, beta, sigma, S, A, alpha, delta)
        HH_guess = np.append(b_sp1_guess, n_s_guess)
        
        sol = opt.root(ne.FOCs, HH_guess, args=foc_args)
        
        b_sp1 = sol.x[0: S-1]
        n_s = sol.x[S-1:]
        euler_errors = sol.fun
        b_s = np.append(0.0, b_sp1)
        
        # use market clearing
        K = ne.get_K(b_s)
        L = ne.get_L(n_s)

        r_prime = ne.get_r(K, L, alpha, delta, A)
        dist = (r - r_prime) ** 2
        iterr += 1
        r = xi * r_prime + (1 - xi) * r
        
        I = ne.get_I(n_s)
        Y = ne.get_Y(n_s)
        C = ne.get_C(n_s)
    success = iterr < max_iter
    return iterr, r, w, b_sp1, K, L, Y, I, C, euler_errors, success
