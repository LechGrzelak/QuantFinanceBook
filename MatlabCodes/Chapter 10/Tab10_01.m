function SVL_Table_Convergence3
clear;close all;clc;warning('off');

StepsYr = 8;
method  = 'AES';
%method  = 'Euler';

%% --Input parameters-- %%
S0 = 1;   %Initial stock price.
M0 = 1;
r  = 0.0; %Initial interest rate. Note that for different r, the code needs a minor corrections
t0 = 0;   %Initial time.
T2 = 5;

%%Market data are generated with the Heston model, here are the "market"
%%parameters
gamma  = 0.95;   %Volatility variance Heston ('vol-vol'). All market parameters.
kappa  = 1.05;   %Reversion rate Heston.
rho    = -0.315;    %Correlation BM's Heston.
vBar   = 0.0855; %Long run variance Heston.
v0     = 0.0945; %Initial variance Heston.
F      = 2*kappa*vBar/(gamma^2); %Feller condition is satisfied for F >= 1.

%%--'Calibrated' MODEL parameters--
%p = 0.05; %Close
p = 0.25; %Moderate
%p = 0.6; %Far off
gammaModel  = (1-p)*gamma;
kappaModel  = (1+p)*kappa;
rhoModel = (1+p)*rho;
vBarModel   = (1-p)*vBar;
v02     = (1+p)*v0;
F2      = 2*kappaModel*vBarModel/(gammaModel^2); %Feller condition is satisfied for F2 >= 1.

%%--Settings--
NoBins    = 20;
NoOfPaths = 5e4; %Monte Carlo: #paths per seed.
NoOfSteps = StepsYr*T2;
dt        = 1/StepsYr;
k         = [0.7 1 1.5];

%% --Define market-- %%
cf = @(u,T)ChFHeston(u, T, kappa,vBar,gamma,rho, v0, r);
Vc = @(t,x)Market(cf,t,x,log(S0));

% Define bump size
bump_T  = 1e-4;
bump_K  = @(T)1e-4;

% Define derivatives
dC_dT   = @(T,K) (Vc(T + bump_T,K) - Vc(T ,K)) /  bump_T;
dC_dK   = @(T,K) (Vc(T,K + bump_K(T)) - Vc(T,K - bump_K(T))) / (2 * bump_K(T));
d2C_dK2 = @(T,K) (Vc(T,K + bump_K(T)) + Vc(T,K-bump_K(T)) - 2*Vc(T,K)) / bump_K(T)^2;

%% --Get price back out of local volatility model-- %%
NoSeeds      = 20; 
ImpliedVol_T2Sum  = 0;
ImpliedVol_T2Mat  = zeros(length(k),NoSeeds);
EuropOptionPriceT2Sum     = 0;

for s =1 : NoSeeds   
    randn('seed',s)
    rand('seed',s)
    %fprintf('seed number = %2.0f',s)   
    t   = t0;
    
    S = S0+zeros(NoOfPaths,1);
    x = log(S0)+zeros(NoOfPaths,1);
    M = M0+zeros(NoOfPaths,1);
    v = v02+zeros(NoOfPaths,1);
    v_old = v;
    
    for i = 1:NoOfSteps
        fprintf('seed number = %.0f and time= %2.2f of t_max = %2.2f',s,t,T2)
        fprintf('\n')
        t_real = t;
        
        if i==1
            t_adj = 1/NoOfSteps;
            t     = t_adj; %Flat extrapolation with respect to time OF maturity.
        end       
               
        nominator = dC_dT(t,S) + r*S.*dC_dK(t,S);
        
        % Note that we need to apply "max" as on some occasions we may have S=0. 
        % This can be improved further however it requires a
        % more in depth computation of the derivatives.
        d2CdK2 = max(abs(d2C_dK2(t,S)),1e-7);
        sigma_det_SLV = sqrt(abs(nominator)./(1/2*S.^2.*d2CdK2));
        
        if t_real ~= 0
            EV = getEV_Better(S,v_new,NoBins);
            sigma_new = sigma_det_SLV./sqrt(EV);
        else
            sigma_new = sigma_det_SLV/sqrt(v0);
            sigma_old = sigma_new;
        end
       
        if strcmp(method,'AES') == 1
            Z_x       = randn(NoOfPaths,1);
            Z_x       = (Z_x-mean(Z_x))/std(Z_x);
            v_new = CIR_Sample(NoOfPaths,kappaModel,gammaModel,vBarModel,0,dt,v_old);
            x = x+r*dt-1/2*sigma_old.^2.*v_old*dt...
                +(rhoModel*sigma_old/gammaModel).*(v_new-v_old-kappaModel*vBarModel*dt+kappaModel*v_old*dt)...
                +sqrt(1-rhoModel^2)*Z_x.*sqrt(sigma_old.^2.*v_old*dt);   
        else
            dW      = randn(NoOfPaths,2);
            dW      = (dW-ones(length(dW(:,1)),1)*mean(dW,1))./(ones(length(dW(:,1)),1)*std(dW));
            dW(:,2) = rhoModel*dW(:,1)+sqrt(1-rhoModel^2)*dW(:,2);
            x   = x+(r-1/2*sigma_old.^2.*v)*dt+sigma_old.*sqrt(v*dt).*dW(:,2);
          
            v       = v + kappaModel*(vBarModel-v)*dt+gammaModel*sqrt(v*dt).*dW(:,1); 
            v       = max(v,0); %Heston scheme according to Lord e.a. [2006].
            v_new   = v;        
        end
        
        M   = M+r*M*dt;
        S   = exp(x);

        %%--Moment matching S--%%
        a2_SLV = (S0/M0-mean(S./M))/mean(1./M); 
        S  = max(S + a2_SLV,1e-4);
        
        if i == 1
            t = t_real;
        end
        
        sigma_old = sigma_new;
        v_old     = v_new;
        t         = t+dt;
        
    end
    Tvec(2) = t;
    EuropOptionPriceT2   = MCprice(k,S,Tvec(2),r);
    
    %% --Get implied volatility-- %%
    ImpliedVol_T2 = ImpliedVolatility('c',EuropOptionPriceT2,k,T2,S0,r,0.3);
       
    %% --Sum up seeds-- %%
    ImpliedVol_T2Sum      = ImpliedVol_T2Sum + ImpliedVol_T2;
    ImpliedVol_T2Mat(:,s) = ImpliedVol_T2(:,1);
    EuropOptionPriceT2Sum         = EuropOptionPriceT2Sum + EuropOptionPriceT2;
    %end
    
end
ImpliedVol_T2 = ImpliedVol_T2Sum / NoSeeds;
EuropOptionPriceT2    = EuropOptionPriceT2Sum / NoSeeds;

%% --Compare Implied volatilities from Market and the Model-- %%

EuropOptionPriceT2_Model  = EuropOptionPriceT2;
EuropOptionPriceT2_Market = Carr_Madan_Call(cf,T2,k',log(S0));%Carr_Madan_Call('ChF_H',k',t0,T2,init,parms,settings,'European'); 

ImpliedVol_T2_Model   = ImpliedVol_T2;
ImpliedVol_T2_Market  = ImpliedVolatility('c',EuropOptionPriceT2_Market,k,T2,S0,r,0.3); 


%% --Results-- %%
Strikes  = k;
AbsError = 100*abs(ImpliedVol_T2_Model-ImpliedVol_T2_Market);
StandardDeviation = std(100*ImpliedVol_T2Mat')' ;

if strcmp(method,'AES') == 1
    fprintf('====== THE AES SCHEME======')    
else
     fprintf('====== THE EULER SCHEME======')    
end
fprintf('====== CASE for dt =%.4f',dt)
fprintf('\n')
for i =1: length(k)
    fprintf('for strike %.2f: abs IV error = %.2f (%.2f)',Strikes(i),AbsError(i),StandardDeviation(i))
    fprintf('\n')
end

stop=1
function sample = CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
sample   = c * random('ncx2',d,kappaBar,[NoOfPaths,1]);

function value = Market(cf,T,x,x0)
value = Carr_Madan_Call(cf,T,x,x0);

function value = Carr_Madan_Call(ChF,T,K,x0)
% Make sure that we don't evaluate at 0
K(K<1e-5)=1e-5;

alpha   = 0.75; %Carr-Madan
c       = 3e2;
N_CM    = 2^12;

eta    = c/N_CM;
b      = pi/eta;
u      = [0:N_CM-1]*eta;
lambda = 2*pi/(N_CM*eta);
i      = complex(0,1);

u_new = u-(alpha+1)*i; %European call option.
cf    = exp(i*u_new*x0).*ChF(u_new,T);
psi   = cf./(alpha^2+alpha-u.^2+i*(2*alpha+1)*u);

SimpsonW         = 3+(-1).^[1:N_CM]-[1,zeros(1,N_CM-1)];
SimpsonW(N_CM)   = 0;
SimpsonW(N_CM-1) = 1;
FFTFun           = exp(i*b*u).*psi.*SimpsonW;
payoff           = real(eta*fft(FFTFun)/3);
strike           = exp(-b:lambda:b-lambda);
payoff_specific  = spline(strike,payoff,K);
value = exp(-log(K)*alpha).*payoff_specific/pi;

function cf=ChFHeston(u, tau, kappa,vBar,gamma,rho, v0, r)
i     = complex(0,1);

% functions D_1 and g
D_1  = sqrt(((kappa -i*rho*gamma.*u).^2+(u.^2+i*u)*gamma^2));
g    = (kappa- i*rho*gamma*u-D_1)./(kappa-i*rho*gamma*u+D_1);    

% complex valued functions A and C
C = (1/gamma^2)*(1-exp(-D_1*tau))./(1-g.*exp(-D_1*tau)).*(kappa-gamma*rho*i*u-D_1);
A = i*u*r*tau + kappa*vBar*tau/gamma^2 * (kappa-gamma*rho*i*u-D_1)-2*kappa*vBar/gamma^2*log((1-g.*exp(-D_1*tau))./(1-g));

% ChF for the Heston model
cf = exp(A + C * v0);

function val = getEV_Better(S,v,NoBins)
if NoBins ~= 1
    mat     = sortrows([S v],1);
    BinSize = floor(length(mat)/NoBins);
    
    first = [mean(mat(1:BinSize,2)) mat(1,1) mat(BinSize,1)];
    last  = [mean(mat((NoBins-1)*BinSize:NoBins*BinSize,2)) mat((NoBins-1)*BinSize,1) mat(NoBins*BinSize,1)];
    val   = mean(mat(1:BinSize,2))*(S>=first(2) & S<first(3));
    
    for i = 2 : NoBins-1
        val = val + mean(mat((i-1)*BinSize:i*BinSize,2))*(S>=mat((i-1)*BinSize,1) & S<mat(i*BinSize,1));
    end
    
    val  = val + last(1)*(S>=last(2) & S<=last(3));
    val(val==0) = last(1);
else
    val = mean(v)*ones(length(S),1);
end

function val = MCprice(Kvec,Smat,T,r)
val = zeros(length(Kvec),length(Smat(1,:)));
for i = 1:length(Kvec)
    payoff = max(0,Smat-Kvec(i));  
    for j = 1:length(Smat(1,:))
        val(i,j) = exp(-r*T)*mean(payoff(:,j));
    end
end

% Exact pricing of European Call/Put option with the Black-Scholes model
function value=BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)
% Black-Scholes Call option price
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function impliedVol = ImpliedVolatility(CP,marketPrice,K,T,S_0,r,initialVol)
    impliedVol= zeros(length(K),1);
    for i = 1:length(K)
        func = @(sigma) (BS_Call_Option_Price(CP,S_0,K(i),sigma,T,r) - marketPrice(i)).^1.0;
        impliedVol(i) = fzero(func,initialVol);
    end