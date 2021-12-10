import SS
import numpy as np

#periods
S = int(100)
# model parameters
alpha = 0.3
delta = 0.1
A = 1.0
sigma = 1.5
beta = 0.8
b_radius= 0.501
v=1.554
l_tilde=1
chi = 1.0 * np.ones(S)

# Make initial guesses
r_guess = 0.1
b_guess = np.array([0, 0])


args=(l_tilde, chi, b_radius, v, b_guess, beta, sigma, S, A, alpha, delta)

iterr, r, w, b_sp1, K, L, Y, I, C, euler_errors, success = SS.SS_solver(r_guess, args)

