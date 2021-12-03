import numpy as np
import matplotlib.pyplot as plt
import numba

'''
------------------------------------------------------------------------
Create grid of current utility values    
------------------------------------------------------------------------
C        = matrix, current consumption (c=w-w')
U        = matrix, current period utility value for all possible
           choices of w and w' (rows are w, columns w')
------------------------------------------------------------------------
'''


def utility_function(size_w, D_grid, r, w, yH, yL, sigma):
    '''
    ------------------------------------------------------------------------
    Create grid of current utility values    
    ------------------------------------------------------------------------
    C        = matrix, current consumption (c=w-w')
    U        = matrix, current period utility value for all possible choices of D and D' (rows are D, columns D')
    ------------------------------------------------------------------------
    '''
    CH = np.zeros((size_w, size_w))
    CL = np.zeros((size_w, size_w))
    
    for i in range(size_w): # loop over w
        for j in range(size_w): # loop over w'
            CH[i, j] = D_grid[i]*(1+r) - D_grid[j] + w*yH # note that if D'>D+wyH, consumption negative
            CL[i, j] = D_grid[i]*(1+r) - D_grid[j] + w*yL
    # replace 0 and negative consumption with a tiny value 
    # This is a way to impose non-negativity on cons
    CH[CH<=0] = 1e-15
    CL[CL<=0] = 1e-15
    
    if sigma == 1:
        UH = np.log(CH)
        UL = np.log(CL)
    else:
        UH = (CH ** (1 - sigma)) / (1 - sigma)
        UL = (CL ** (1 - sigma)) / (1 - sigma)
    UH[CH<0] = -9999999
    UL[CL<0] = -9999999
    return UH, UL


@numba.jit()
def valuefuction_iter(VFdist, VFtol, VFiter, VFmaxiter, size_w, VstoreH, VstoreL, VmatH, VmatL, VH, VL, UH, UL, PHH, PHL, PLL, PLH, beta):
    while VFdist > VFtol and VFiter < VFmaxiter:
        for i in range(size_w): # loop over D
            for j in range(size_w): # loop over D'
                VmatH[i, j] = UH[i, j] + beta * (PHL*VL[j] + PHH*VH[j])
                VmatL[i, j] = UL[i, j] + beta * (PLL*VL[j] + PLH*VH[j])
    
        VstoreH[:, VFiter] = VH.reshape(size_w,) # store value function at each iteration for graphing later
        VstoreL[:, VFiter] = VL.reshape(size_w,)
        TVH = VmatH.max(1) # apply max operator to Vmat (to get V(w))
        TVL = VmatL.max(1)
        
        PFH = np.argmax(VmatH, axis=1)
        PFL = np.argmax(VmatL, axis=1)
        
        VFdistH = (np.absolute(VH - TVH)).max()  # check distance
        VFdistL = (np.absolute(VL - TVL)).max()  # check distance
        VFdist= max(VFdistH, VFdistL)    
        VH = TVH
        VL = TVL
        VFiter += 1 
        
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')
    
    VFH = VH # solution to the functional equation
    VFL = VL # solution to the functional equation
    
    return VFH, VFL, PFH, PFL

