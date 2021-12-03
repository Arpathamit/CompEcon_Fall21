# Import packages
import numpy as np
import matplotlib.pyplot as plt
from functions import utility_function
from functions import valuefuction_iter


#parameters
beta = 0.95
sigma = 1.0
r=0.1
w=1

#probabilities
PHH=0.8;  #probability of starting at high shock and stay at high shock
PHL=0.2;
PLL=0.5;
PLH=0.5;

#high income and low income 
yH=1;     #high shock to the HH's income
yL=.1;    #low shock to the HH's income


'''
------------------------------------------------------------------------
Create Grid for State Space    
------------------------------------------------------------------------
lb_w      = scalar, lower bound of debt grid
ub_w      = scalar, upper bound of debt grid 
size_w    = integer, number of grid points in debt state space
w_grid    = vector, size_w x 1 vector of debt grid points 
------------------------------------------------------------------------
'''
#let the grid consist of 60 equally-spaced points on the interval [0, 5]
lb_w = -10
ub_w = 10
size_w = 100  # Number of grid points
D_grid = np.linspace(lb_w, ub_w, size_w)


'''
------------------------------------------------------------------------
Value Function Iteration    
------------------------------------------------------------------------
VFtol     = scalar, tolerance required for value function to converge
VFdist    = scalar, distance between last two value functions
VFmaxiter = integer, maximum number of iterations for value function
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of w and w'
Vstore    = matrix, stores V at each iteration 
VFiter    = integer, current iteration number
TV        = vector, the value function after applying the Bellman operator
PF        = vector, indicies of choices of w' for all w 
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''
VFtol = 1e-4
VFdist = 7.0 
VFmaxiter = 3000
VH = np.zeros(size_w) # initial guess at value function
VL = np.zeros(size_w)
#Vmat = np.zeros((size_w, size_w)) # initialize Vmat matrix
VmatH = np.zeros((size_w, size_w))
VmatL = np.zeros((size_w, size_w))
Vstore = np.zeros((size_w, VFmaxiter)) #initialize Vstore array
VstoreH = np.zeros((size_w, VFmaxiter))
VstoreL = np.zeros((size_w, VFmaxiter))
VFiter = 1 


UH, UL = utility_function(size_w, D_grid, r, w, yH, yL, sigma)

VFH, VFL, PFH, PFL = valuefuction_iter(VFdist, VFtol, VFiter, VFmaxiter, size_w, VstoreH, VstoreL, VmatH, VmatL, VH, VL, UH, UL, PHH, PHL, PLL, PLH, beta)

############################################################################


'''
-----------------------------------------------
Find consumption and savings policy functions   
------------------------------------------------------------------------
optW  = vector, the optimal choice of w' for each w
optC  = vector, the optimal choice of c for each c
------------------------------------------------------------------------
'''
optDH = D_grid[PFH] # tomorrow's optimal debt (savings function)
optCH = D_grid - optDH # optimal consumption - get consumption through the transition eqn

optDL = D_grid[PFL] # tomorrow's optimal debt (savings function)
optCL = D_grid - optDL # optimal consumption - get consumption through the transition eqn


# Plot value function 
plt.figure()
fig, ax = plt.subplots()
ax.scatter(D_grid[1:], VFH[1:], label='Starting with high wage')
ax.scatter(D_grid[1:], VFL[1:], label='Starting with low wage')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of debt')
plt.ylabel('Value Function')
plt.title('Value Function - debt')
plt.show()


#Plot optimal consumption rule as a function of debt
plt.figure()
fig, ax = plt.subplots()
ax.plot(D_grid[3:], optCH[3:], label='Consumption for H')
ax.plot(D_grid[3:], optCL[3:], label='Consumption for L')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of debt')
plt.ylabel('Optimal Consumption')
plt.title('Policy Function, consumption - debt')
plt.show()


#Plot debt to leave rule as a function of debt
plt.figure()
fig, ax = plt.subplots()
ax.plot(D_grid[1:], optDH[1:], label='saving if starting with high wage')
ax.plot(D_grid[1:], optDL[1:], label='saving if starting with low wage')
ax.plot(D_grid[1:], D_grid[1:], '--', label='45 degree line')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of debt')
plt.ylabel('Optimal transfer')
plt.title('Policy Function, savings -  debt')
plt.show()