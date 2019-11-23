#%%
"""
Importance Sampling exercise
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Global variable 

a = 3

# Benchmark

def ExactValue():
    
    val = 1.0- st.norm.cdf(a,0,1)
    print("Reference value is equal to %.2f" %val)
    return val   
        
# Monte Carlo simulation of the problem

def MonteCarloSimulation():    

    # Specification of the grid for the number of samples

    gridN = range(1000,100000,100)
    
    solution = np.zeros([len(gridN)])
    solutionCase_1 = np.zeros([len(gridN)])
    solutionCase_2 = np.zeros([len(gridN)])
    solutionCase_3 = np.zeros([len(gridN)])
    solutionCase_4 = np.zeros([len(gridN)])
    
    # Lambda expressions for Radon-Nikodym derivative

    L1 = lambda x: st.norm.pdf(x,0.0,1.0)/st.uniform.pdf(x,0.0,1.0)
    L2 = lambda x: st.norm.pdf(x,0.0,1.0)/st.uniform.pdf(x,0.0,4.0)
    L3 = lambda x: st.norm.pdf(x,0.0,1.0)/st.norm.pdf(x,0.0,0.5)
    L4 = lambda x: st.norm.pdf(x,0.0,1.0)/st.norm.pdf(x,0.0,3.0)
    
    #Generate results 

    for idx, N in enumerate(gridN):

        # Direct Monte Carlo simulation

        X = np.random.normal(0.0,1.0,N)
        RV = X>a
        solution[idx] = np.mean(RV)
        
        # Case 1, uniform (0,1)

        Y_temp = np.random.uniform(0.0,1.0,N)
        RV = (Y_temp>a) * 1.0 * L1(Y_temp)
        solutionCase_1[idx] = np.mean(RV)
        
        # Case 2, uniform (0,4)

        Y_temp = np.random.uniform(0.0,4.0,N)
        RV = (Y_temp>a) * 1.0 * L2(Y_temp)
        solutionCase_2[idx] = np.mean(RV)
        
        # Case 3, normal (0,0.5)

        Y_temp = np.random.normal(0.0,0.5,N)
        RV = (Y_temp>a) * 1.0 * L3(Y_temp)
        solutionCase_3[idx] = np.mean(RV)
        
        # Case 4, normal (0,3)

        Y_temp = np.random.normal(0.0,3.0,N)
        RV = (Y_temp>a) * 1.0 * L4(Y_temp)
        solutionCase_4[idx] = np.mean(RV)
        
    plt.plot(gridN,ExactValue() * np.ones(len(gridN)))
    plt.plot(gridN,solution)
    plt.plot(gridN,solutionCase_1)
    plt.plot(gridN,solutionCase_2)
    plt.plot(gridN,solutionCase_3)
    plt.plot(gridN,solutionCase_4)
    plt.xlabel("Number of Samples")
    plt.ylabel("Estimated value")
    plt.grid()
    plt.legend(["exact","Monte Carlo","Case 1", "Case 2", "Case 3", "Case 4"])
