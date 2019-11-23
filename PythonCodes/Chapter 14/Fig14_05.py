#%%
"""
Created on Feb 10 2019
Ploting of the rates in positive and negative rate environment
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
   
def mainCalculation():
    time = np.linspace(0.1,30,50)
    Rates2008 = [4.4420,4.4470,4.3310,4.2520,4.2200,4.2180,4.2990,4.3560,4.4000,\
    4.4340,4.4620,4.4840,4.5030,4.5190,4.5330,4.5450,4.5550,4.5640,4.5720,\
    4.5800,4.5860,4.5920,4.5980,4.6030,4.6070,4.6110,4.6150,4.6190,4.6220,\
    4.6250,4.6280,4.6310,4.6340,4.6360,4.6380,4.6400,4.6420,4.6440,4.6460,4.6480,\
    4.6490,4.6510,4.6520,4.6540,4.6550,4.6560,4.6580,4.6590,4.6600,4.6610]

    Rates2017 = [-0.726,-0.754,-0.747,-0.712,-0.609,-0.495,-0.437,-0.374,-0.308,\
    -0.242,-0.177,-0.113,-0.0510,0.00900,0.0640,0.115,0.163,0.208,0.250,0.288,\
    0.323,0.356,0.386,0.414,0.439,0.461,0.482,0.501,0.519,0.535,0.550,0.564,\
    0.577,0.588,0.598,0.608,0.617,0.625,0.632,0.640,0.646,0.652,0.658,0.663,\
    0.668,0.673,0.678,0.682,0.686,0.690]
    
    plt.figure(1)
    plt.plot(time,Rates2008)
    plt.grid()
    plt.title('Interest Rates, EUR1M, 2008')
    plt.xlabel('time in years')
    plt.ylabel('yield')

    plt.figure(2)
    plt.plot(time,Rates2017)
    plt.grid()
    plt.title('Interest Rates, EUR1M, 2017')
    plt.xlabel('time in years')
    plt.ylabel('yield')    
    
mainCalculation()