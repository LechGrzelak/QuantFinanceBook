#%%
"""
Created on Apr 24 2019
The Heston-SLV model, Monte Carlo simulation. 
Note that because of the model complexity Matlab's implementation outperforms
in terms of speed one presented here. 
@author: Lech A. Grzelak
"""
import numpy as np
import enum 
from scipy.interpolate import CubicSpline

# set i= imaginary number
i   = np.complex(0.0,1.0)

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)  
        
    # reshape K to a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    #assigning i=sqrt(-1)
    i = np.complex(0.0,1.0) 
    x0 = np.log(S0 / K)   
    
    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # sumation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for Put Prices  
    H_k = CallPutCoefficients(CP,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))     
    return value

# Determine coefficients for Put Prices 
def CallPutCoefficients(CP,a,b,k):
    if CP==OptionType.CALL:                  
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Chi_k - Psi_k)  
    elif CP==OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    
    return H_k    

def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi 
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * 
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k 
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value

def ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))*(kappa-gamma*rho*i*u-D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1-g(u)*np.exp(-D1(u)*tau))/(1-g(u)))
    # Characteristic function for the Heston's model    
    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf 

def  Carr_Madan_Call(ChF,T,K,x0):
    # Make sure that we don't evaluate at 0
    #K(K<1e-5)=1e-5
    
    alpha   = 0.75 
    c       = 300.0
    N_CM    = int(np.power(2.0,12))
    
    eta    = c/N_CM
    b      = np.pi/eta
    u      = np.linspace(0,N_CM-1,N_CM)*eta #[0:N_CM-1]*eta
    lambd  = 2.0*np.pi/(N_CM*eta)
    i      = np.complex(0,1)
    
    u_new = u-(alpha+1.0)*i #European call option.
    
    cf    = np.exp(i*u_new*x0)*ChF(u_new)
    psi   = cf/(alpha**2.0+alpha-u**2.0+i*(2.0*alpha+1.0)*u)
    
    hlp = np.zeros([int(N_CM)])
    hlp[0] =1
    SimpsonW         = 3.0+np.power(-1,np.linspace(1,N_CM,N_CM))-hlp
    SimpsonW[N_CM-1] = 0.0
    SimpsonW[N_CM-2] = 1.0
    FFTFun           = np.exp(i*b*u)*psi*SimpsonW
    payoff           = np.real(eta*np.fft.fft(FFTFun)/3.0)
    strike           = np.exp(np.linspace(-b,b,N_CM)-lambd)
    payoff_interpol  = CubicSpline(strike,payoff,bc_type='clamped')
    payoff_specific  = payoff_interpol(K)
    
    value = np.exp(-np.log(K)*alpha)*payoff_specific/np.pi
    
    return value

def CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    sample = c* np.random.noncentral_chisquare(delta,kappaBar,NoOfPaths)
    return  sample

def getEVBinMethod(S,v,NoOfBins):
    if (NoOfBins != 1):
        mat  = np.transpose(np.array([S,v]))
       
        # sort all the rows according to the first column
        val = mat[mat[:,0].argsort()]
        
        binSize = int((np.size(S)/NoOfBins))
         
        expectation = np.zeros([np.size(S),2])

        for i in range(1,binSize-1):
            sample = val[(i-1)*binSize:i*binSize,1]
            expectation[(i-1)*binSize:i*binSize,0] =val[(i-1)*binSize:i*binSize,0]
            expectation[(i-1)*binSize:i*binSize,1] =np.mean(sample)
        return expectation

def EUOptionPriceFromMCPathsGeneralized(CP,S,K,T,r):
    # S is a vector of Monte Carlo samples at T
    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(k-S,0.0))
    return result

def mainCalculation():
    
    T2      = 5.0
    t0      = 0.0
    StepsYr = 8;
    method  = 'AES'
    #method  = 'Euler'
    
    #--Settings--
    NoBins    = 20
    NoOfPaths = 50 #paths per seed.
    NoOfSteps = int(StepsYr*T2)
    dt        = 1/StepsYr
    k         = np.array([0.7, 1.0, 1.5])
    
    # Market parameters
    gamma  = 0.95   #Volatility variance Heston ('vol-vol'). All market parameters.
    kappa  = 1.05  #Reversion rate Heston.
    rho    = -0.315   #Correlation BM's Heston.
    vBar   = 0.0855#Long run variance Heston.
    v0     = 0.0945#Initial variance Heston.
    S0     = 1.0
    x0     = np.log(S0)
    r      = 0.0
        
    # Number of expansion terms for the COS method
    N = 5000
    # COS method settings
    L = 8
    
    # Model parameters
    p = 0.25 #Moderate
    gammaModel  = (1-p)*gamma
    kappaModel  = (1+p)*kappa
    rhoModel = (1+p)*rho
    vBarModel   = (1-p)*vBar
    v02     = (1+p)*v0
    
    # Define market
    cf  = ChFHestonModel(r,T2,kappa,gamma,vBar,v0,rho)
    Vc  = lambda T,K: CallPutOptionPriceCOSMthd(cf, OptionType.CALL, S0, r, T, K, N, L)
    
    # Define derivatives    
    # Define bump size
    bump_T  = 1e-4
    bump_K  = lambda T:1e-4

    dC_dT   = lambda T,K: (Vc(T + bump_T,K) - Vc(T ,K)) /  bump_T
    dC_dK   = lambda T,K: (Vc(T,K + bump_K(T)) - Vc(T,K - bump_K(T))) / (2.0 * bump_K(T))
    d2C_dK2 = lambda T,K: (Vc(T,K + bump_K(T)) + Vc(T,K-bump_K(T)) - 2.0*Vc(T,K)) / bump_K(T)**2.0
    
    ## --Get price back out of local volatility model-- %%
    NoSeeds      = 2
    
    for s in range(1, NoSeeds+1  ):
        np.random.seed(s)    
        t   = t0
        S = S0+np.zeros([NoOfPaths,1])
        x = np.log(S0)+np.zeros([NoOfPaths,1])
        M = 1.0+np.zeros([NoOfPaths,1])
        v = v02+np.zeros([NoOfPaths,1])
        v_new = v
        v_old = v
    
        for i in range(1,NoOfSteps):
            print('seed number = {0} and time= {1} of t_max = {2}'.format(s,t,T2))
            t_real = t;
        
            if i==1:
                t_adj = 1/NoOfSteps
                t     = t_adj; #Flat extrapolation with respect to time OF maturity.                
               
            nominator = dC_dT(t,S) + r*S*dC_dK(t,S)
        
            # Note that we need to apply "max" as on some occasions we may have S=0. 
            # This can be improved further however it requires a
            # more in depth computation of the derivatives.
            d2CdK2 = np.maximum(np.abs(d2C_dK2(t,S)),1e-7)
            sigma_det_SLV = np.sqrt(np.abs(nominator)/(0.5*S**2.0*d2CdK2))
        
            if t_real != 0:
                EV = getEVBinMethod(S[:,0],v_new[:,0],NoBins)
                EV = EV[:,1].reshape([NoOfPaths,1])
                sigma_new = sigma_det_SLV/np.sqrt(EV)
            else:
                sigma_new = sigma_det_SLV/np.sqrt(v0)
                sigma_old = sigma_new
            
           
            if method=='AES':
                Z_x       = np.random.normal(0.0,1.0,[NoOfPaths,1])
                Z_x       = (Z_x-np.mean(Z_x))/np.std(Z_x)
                v_new[:,0]     = np.array(CIR_Sample(NoOfPaths,kappaModel,gammaModel,vBarModel,0,dt,v_old[:,0]))
                x = x+r*dt-1/2*sigma_old**2.0*v_old*dt+(rhoModel*sigma_old/gammaModel)*(v_new-v_old-kappaModel*vBarModel*dt+kappaModel*v_old*dt)+np.sqrt(1-rhoModel**2.0)*Z_x*np.sqrt(sigma_old**2.0*v_old*dt)   
            else:
                #dW      = randn(NoOfPaths,2)
                #dW      = (dW-ones(length(dW(:,1)),1)*mean(dW,1))./(ones(length(dW(:,1)),1)*std(dW))
                #dW(:,2) = rhoModel*dW(:,1)+sqrt(1-rhoModel^2)*dW(:,2)
                Z1 = np.random.normal(0.0,1.0,[NoOfPaths,1])
                Z2 = np.random.normal(0.0,1.0,[NoOfPaths,1])
                if NoOfPaths > 1:
                    Z1 = (Z1 - np.mean(Z1)) / np.std(Z1)
                    Z2 = (Z2 - np.mean(Z2)) / np.std(Z2)
                    Z2 = rhoModel * Z1 + np.sqrt(1.0-rhoModel**2)*Z2 
                 
                x   = x+(r-0.5*sigma_old**2.0*v)*dt+sigma_old*np.sqrt(v*dt)*Z1
              
                v       = v + kappaModel*(vBarModel-v)*dt+gammaModel*np.sqrt(v*dt)*Z2
                v       = np.maximum(v,0) #Heston scheme according to Lord e.a. [2006].
                v_new   = v        
            
            
            M   = M+r*M*dt
            S   = np.exp(x)
    
            #--Moment matching S--%%
            a2_SLV = (S0-np.mean(S/M))/np.mean(1/M) 
            S  = np.maximum(S + a2_SLV,1e-4)
            
            if i == 1:
                t = t_real
            
            sigma_old = sigma_new
            v_old     = v_new
            t         = t+dt
        OptPrice   = EUOptionPriceFromMCPathsGeneralized(OptionType.CALL,S[:,-1],k*S0,T2,0.0)
        print('option prices for Heston SLV model are {0}'.format(OptPrice))
    
mainCalculation()