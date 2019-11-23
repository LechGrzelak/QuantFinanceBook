function SVL_Forward_ImpVols
clear;close all;clc;warning('off');

StepsYr = 10;
method  = 'AES';
%method  = 'Euler';

%% --Input parameters-- %%
S0 = 1;   %Initial stock price.
M0 = 1;
r  = 0.0; %Initial interest rate. Note that for different r, the code needs a minor corrections
t0 = 0;   %Initial time.
T1 = 2;
T2 = 5;

%%Market data are generated with the Heston model, here are the "market"
%%parameters
gamma  = 0.5; %Volatility variance Heston ('vol-vol'). All market parameters.
kappa  = 0.3; %Reversion rate Heston.
rho = -0.6; %Correlation BM's Heston.
vBar   = 0.05; %Long run variance Heston.
v0     = 0.04; %Initial variance Heston.

%%--'Calibrated' MODEL parameters--
%p = 0.05; %Close
p = 0.25; %Moderate
%p = 0.6; %Far off
gammaModel  = (1-p)*gamma;
kappaModel  = (1+p)*kappa;
rhoModel = (1+p)*rho;
vBarModel   = (1-p)*vBar;
v0Model     = (1+p)*v0;

%%--Settings--
NoBins    = 20;
NoOfPaths = 5e4; %Monte Carlo: #paths per seed.
NoOfSteps = StepsYr*T2;
dt        = 1/StepsYr;
delta   = -3.5:0.5:3.5;
k       = S0*exp(0.1*sqrt(T2)*delta);
doneFlag = 0;
%% --Define market-- %%
cf = @(u,T)ChFHeston(u, T, kappa,vBar,gamma,rho, v0, r);
Vc = @(t,x)Market(cf,t,x,log(S0));

%% --Define Heston's model that is precalibrated to the market
cfHeston     = @(u,T)ChFHeston(u, T, kappaModel,vBarModel,gammaModel,rhoModel, v0Model, r);
cfHestonFrwd = @(u) ChFHestonForward(u, T1, T2, kappaModel,vBarModel,gammaModel,rhoModel, v0Model ,r);
optPriceHestonFrwd = CallPutOptionPriceCOSMthd_FrwdStart(cfHestonFrwd,'c',r,T1,T2,k'-1,2000,5);

% Define bump size
bump_T  = 1e-4;
bump_K  = @(T)1e-4;

% Define derivatives
dC_dT   = @(T,K) (Vc(T + bump_T,K) - Vc(T ,K)) /  bump_T;
dC_dK   = @(T,K) (Vc(T,K + bump_K(T)) - Vc(T,K - bump_K(T))) / (2 * bump_K(T));
d2C_dK2 = @(T,K) (Vc(T,K + bump_K(T)) + Vc(T,K-bump_K(T)) - 2*Vc(T,K)) / bump_K(T)^2;

%% --Get price back out of local volatility model-- %%
NoSeeds      = 1; 
ImpliedVol_T2Sum     = 0;
ImpliedVol_T2Sum_LV     = 0;
ImpliedVol_T2Mat     = zeros(length(k),NoSeeds);
ImpliedVol_T2Mat_LV  = zeros(length(k),NoSeeds);
EuropOptionPriceT2Sum     = 0;
EuropOptionPriceT2Sum_LV     = 0;

for s =1 : NoSeeds   
    randn('seed',s)
    rand('seed',s)
    %fprintf('seed number = %2.0f',s)   
    t   = t0;
    
    S = S0+zeros(NoOfPaths,1);
    S_LV    = S;
    x = log(S0)+zeros(NoOfPaths,1);
    M = M0+zeros(NoOfPaths,1);
    v = v0Model+zeros(NoOfPaths,1);
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
        sigma_det = sqrt(abs(nominator)./(1/2*S.^2.*d2CdK2));
        
        % Pure local volatility setting
        nominator_LV = dC_dT(t,S_LV) + r*S.*dC_dK(t,S_LV);
        d2CdK2_LV = max(abs(d2C_dK2(t,S_LV)),1e-7);
        sigma_det_LV = sqrt(abs(nominator_LV)./(1/2*S_LV.^2.*d2CdK2_LV));
        
        if t_real ~= 0
            EV = getEV_Better(S,v_new,NoBins);
            sigma_new = sigma_det./sqrt(EV);
        else
            sigma_new = sigma_det/sqrt(v0);
            sigma_old = sigma_new;
        end
                      
        if strcmp(method,'AES') == 1
            Z_x       = randn(NoOfPaths,1);
            Z_x       = (Z_x-mean(Z_x))/std(Z_x);
            v_new     = CIR_Sample(NoOfPaths,kappaModel,gammaModel,vBarModel,0,dt,v_old);
            x = x+r*dt-1/2*sigma_old.^2.*v_old*dt...
                +(rhoModel*sigma_old/gammaModel).*(v_new-v_old-kappaModel*vBarModel*dt+kappaModel*v_old*dt)...
                +sqrt(1-rhoModel^2)*Z_x.*sqrt(sigma_old.^2.*v_old*dt);   
            % Pure local volatiliy model
            S_LV = S_LV.*exp((r-1/2*sigma_det_LV.^2)*dt+sigma_det_LV.*sqrt(dt).*Z_x);
        else
%             dW      = randn(NoOfPaths,2);
%             dW      = (dW-ones(length(dW(:,1)),1)*mean(dW,1))./(ones(length(dW(:,1)),1)*std(dW));
%             dW(:,2) = rhoModel*dW(:,1)+sqrt(1-rhoModel^2)*dW(:,2);
            Z_x       = randn(NoOfPaths,1);
            Z_v       = randn(NoOfPaths,1);
            Z_x       = (Z_x-mean(Z_x))/std(Z_x);
            Z_v       = (Z_v-mean(Z_v))/std(Z_v);
            Z_v       = rhoModel*Z_x + sqrt(1-rhoModel^2)*Z_v;
            
            x   = x+(r-1/2*sigma_old.^2.*v)*dt+sigma_old.*sqrt(v*dt).*Z_x;
          
            v       = v + kappaModel*(vBarModel-v)*dt+gammaModel*sqrt(v*dt).*Z_v; 
            v       = max(v,0); %Heston scheme according to Lord e.a. [2006].
            v_new   = v;        
            
            % Pure local volatiliy model
            S_LV = S_LV.*exp((r-1/2*sigma_det_LV.^2)*dt+sigma_det_LV.*sqrt(dt).*Z_x);
        end
        
        M   = M+r*M*dt;
        S   = exp(x);

        %%--Moment matching S--%%
        a2_SLV = (S0/M0-mean(S./M))/mean(1./M); 
        S  = max(S + a2_SLV,1e-4);
        
        a2_LV = (S0/M0-mean(S_LV./M))/mean(1./M); 
        S_LV  = max(S_LV + a2_LV,1e-4);
        
        if i == 1
            t = t_real;
        end
        
        sigma_old = sigma_new;
        v_old     = v_new;
        t         = t+dt;
        
        % Store of S and S_LV for T1
         if (abs(t-T1-dt)<(1+dt)*dt && doneFlag == 0)
             S_T1 = S;
             S_T1_LV = S_LV;
             doneFlag = 1;
             Tvec(1)=t;
         end        
    end
    Tvec(2) = t;
    
    % European-call/put option
    EuropOptionPriceT2      = MCprice(k,S,Tvec(2),r);
    EuropOptionPriceT2_LV   = MCprice(k,S_LV,Tvec(2),r);
    
    % Forward start option
    FrwdStartOpt    = MCprice(k,S./S_T1,Tvec(2),r);
    FrwdStartOpt_LV = MCprice(k,S_LV./S_T1_LV,Tvec(2),r);
        
    
    %% --Get implied volatility-- %%
    ImpliedVol_T2           = ImpliedVolatility('c',EuropOptionPriceT2,k,T2,S0,r,0.3);
    ImpliedVol_T2_LV        = ImpliedVolatility('c',EuropOptionPriceT2_LV,k,T2,S0,r,0.3);
    ImpliedVol_FrwdStart    = ImpliedVolatilityFrwdStart(FrwdStartOpt,k-1,T1,T2,r,0.3);
    ImpliedVol_FrwdStart_LV = ImpliedVolatilityFrwdStart(FrwdStartOpt_LV,k-1,T1,T2,r,0.3);
    ImpliedVol_FrwdStart_Heston = ImpliedVolatilityFrwdStart(optPriceHestonFrwd,k-1,T1,T2,r,0.3);
    
    figure(1)
    plot(k,ImpliedVol_FrwdStart)
    hold on
    plot(k,ImpliedVol_FrwdStart_LV)
    plot(k,ImpliedVol_FrwdStart_Heston)
    
    %% --Sum up seeds-- %%
    ImpliedVol_T2Sum      = ImpliedVol_T2Sum + ImpliedVol_T2;
    ImpliedVol_T2Mat(:,s) = ImpliedVol_T2;
    EuropOptionPriceT2Sum         = EuropOptionPriceT2Sum + EuropOptionPriceT2;
    
    ImpliedVol_T2Sum_LV      = ImpliedVol_T2Sum_LV + ImpliedVol_T2_LV;
    ImpliedVol_T2Mat_LV(:,s) = ImpliedVol_T2_LV;
    EuropOptionPriceT2Sum_LV         = EuropOptionPriceT2Sum_LV + EuropOptionPriceT2_LV;
    %end
    
end
ImpliedVol_T2 = ImpliedVol_T2Sum / NoSeeds;
EuropOptionPriceT2    = EuropOptionPriceT2Sum / NoSeeds;

ImpliedVol_T2_LV = ImpliedVol_T2Sum_LV / NoSeeds;
EuropOptionPriceT2_LV    = EuropOptionPriceT2Sum_LV / NoSeeds;

%% --Compare Implied volatilities from Market and the Model-- %%

EuropOptionPriceT2_Model  = EuropOptionPriceT2;
EuropOptionPriceT2_Market = Carr_Madan_Call(cf,T2,k',log(S0));
EuropOptionPriceT2_Heston = Carr_Madan_Call(cfHeston,T2,k',log(S0));

ImpliedVol_T2_Model   = ImpliedVol_T2;
ImpliedVol_T2_Market  = ImpliedVolatility('c',EuropOptionPriceT2_Market,k,T2,S0,r,0.3); 
ImpliedVol_T2_Heston  = ImpliedVolatility('c',EuropOptionPriceT2_Heston,k,T2,S0,r,0.3); 

%% --Results-- %%
MakeFigure(k, [ImpliedVol_T2_Market,ImpliedVol_T2_LV,ImpliedVol_T2_Model,ImpliedVol_T2_Heston], {'Market','LV','SLV','Heston'},'Implied volatility at T2')

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
    
%Forward start Black-Scholes option price
function value = BS_Call_Option_Price_FrwdStart(K,sigma,T1,T2,r)
K = K + 1.0;
tau = T2 - T1;
d1    = (log(1.0 ./ K) + (r + 0.5 * sigma^2)* tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
value = exp(-r*T1) * normcdf(d1) - normcdf(d2) .* K * exp(-r * T2);

function impliedVol = ImpliedVolatilityFrwdStart(marketPrice,K,T1,T2,r,initialVol)
    impliedVol= zeros(length(K),1);
    for i = 1:length(K)
        func = @(sigma) (BS_Call_Option_Price_FrwdStart(K(i),sigma,T1,T2,r) - marketPrice(i)).^1.0;
        impliedVol(i) = fzero(func,initialVol);
    end

function value = ChFHestonForward(u, T_1, T_2, kappa,vBar,gamma,rho, v0 ,r)
i           = complex(0,1);
tau         = T_2-T_1;
c_bar       = @(t)gamma^2/(4*kappa)*(1-exp(-kappa*t));
d_bar       = 4*kappa*vBar/gamma^2;
kappa_bar  = @(t)4*kappa*v0*exp(-kappa*t)/(gamma^2*(1-exp(-kappa*t)));

% functions D_1 and g
D_1  = sqrt(((kappa -i*rho*gamma.*u).^2+(u.^2+i*u)*gamma^2));
g    = (kappa- i*rho*gamma*u-D_1)./(kappa-i*rho*gamma*u+D_1);    

% complex valued functions A and C
C = (1/gamma^2)*(1-exp(-D_1*tau))./(1-g.*exp(-D_1*tau)).*(kappa-gamma*rho*i*u-D_1);
A = i*u*r*tau + kappa*vBar*tau/gamma^2 * (kappa-gamma*rho*i*u-D_1)-2*kappa*vBar/gamma^2*log((1-g.*exp(-D_1*tau))./(1-g));

% Coefficients used in the forward-ChF
a_1 = C*c_bar(T_1)*kappa_bar(T_1)./(1-2*C*c_bar(T_1));
a_2 = (1./(1-2*C*c_bar(T_1))).^(0.5*d_bar);

% Forward-starting characteristic function
value = exp(A + a_1) .* a_2;

function value = CallPutOptionPriceCOSMthd_FrwdStart(cf,CP,r,T1,T2,K,N,L)
i = complex(0,1);
% cf   - characteristic function as a functon, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - interest rate (constant)
% tau  - time to maturity
% K    - vector of strikes
% N    - Number of expansion terms
% L    - size of truncation domain (typ.:L=8 or L=10)  

% Adjust strike
K = K + 1.0;

% Set up tau and initial x0
tau = T2-T1;
x0  = log(1.0 ./ K);   

% Truncation domain
a = 0 - L * sqrt(tau); 
b = 0 + L * sqrt(tau);

k = 0:N-1;              % row vector, index for expansion terms
u = k * pi / (b - a);   % ChF arguments

H_k = CallPutCoefficients(CP,a,b,k);
temp    = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);     % adjust the first element by 1/2

mat = exp(i * (x0 - a) * u);  % matrix-vector manipulations

% Final output
value = exp(-r * T2) * K .* real(mat * temp);


function value = CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L)
i = complex(0,1);
% cf   - characteristic function as a functon, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - interest rate (constant)
% tau  - time to maturity
% K    - vector of strikes
% N    - Number of expansion terms
% L    - size of truncation domain (typ.:L=8 or L=10)  

x0 = log(S0 ./ K);   

% Truncation domain
a = 0 - L * sqrt(tau); 
b = 0 + L * sqrt(tau);

k = 0:N-1;              % row vector, index for expansion terms
u = k * pi / (b - a);   % ChF arguments

H_k = CallPutCoefficients(CP,a,b,k);
temp    = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);     % adjust the first element by 1/2

mat = exp(i * (x0 - a) * u);  % matrix-vector manipulations

% Final output
value = exp(-r * tau) * K .* real(mat * temp);

% Coefficients H_k for the COS method
function H_k = CallPutCoefficients(CP,a,b,k)
    if lower(CP) == 'c' || CP == 1
        c = 0;
        d = b;
        [Chi_k,Psi_k] = Chi_Psi(a,b,c,d,k);
         if a < b && b < 0.0
            H_k = zeros([length(k),1]);
         else
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k);
         end
    elseif lower(CP) == 'p' || CP == -1
        c = a;
        d = 0.0;
        [Chi_k,Psi_k]  = Chi_Psi(a,b,c,d,k);
         H_k = 2.0 / (b - a) * (- Chi_k + Psi_k);       
    end

function [chi_k,psi_k] = Chi_Psi(a,b,c,d,k)
    psi_k        = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
    psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
    psi_k(1)     = d - c;
    
    chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2); 
    expr1 = cos(k * pi * (d - a)/(b - a)) * exp(d)  - cos(k * pi... 
                  * (c - a) / (b - a)) * exp(c);
    expr2 = k * pi / (b - a) .* sin(k * pi * ...
                        (d - a) / (b - a))   - k * pi / (b - a) .* sin(k... 
                        * pi * (c - a) / (b - a)) * exp(c);
    chi_k = chi_k .* (expr1 + expr2);

function MakeFigure(X1, YMatrix1, argLegend,titleIn)
%CREATEFIGURE(X1,YMATRIX1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 16-Jan-2012 15:26:40

% Create figure
figure1 = figure('InvertHardcopy','off',...
    'Colormap',[0.061875 0.061875 0.061875;0.06875 0.06875 0.06875;0.075625 0.075625 0.075625;0.0825 0.0825 0.0825;0.089375 0.089375 0.089375;0.09625 0.09625 0.09625;0.103125 0.103125 0.103125;0.11 0.11 0.11;0.146875 0.146875 0.146875;0.18375 0.18375 0.18375;0.220625 0.220625 0.220625;0.2575 0.2575 0.2575;0.294375 0.294375 0.294375;0.33125 0.33125 0.33125;0.368125 0.368125 0.368125;0.405 0.405 0.405;0.441875 0.441875 0.441875;0.47875 0.47875 0.47875;0.515625 0.515625 0.515625;0.5525 0.5525 0.5525;0.589375 0.589375 0.589375;0.62625 0.62625 0.62625;0.663125 0.663125 0.663125;0.7 0.7 0.7;0.711875 0.711875 0.711875;0.72375 0.72375 0.72375;0.735625 0.735625 0.735625;0.7475 0.7475 0.7475;0.759375 0.759375 0.759375;0.77125 0.77125 0.77125;0.783125 0.783125 0.783125;0.795 0.795 0.795;0.806875 0.806875 0.806875;0.81875 0.81875 0.81875;0.830625 0.830625 0.830625;0.8425 0.8425 0.8425;0.854375 0.854375 0.854375;0.86625 0.86625 0.86625;0.878125 0.878125 0.878125;0.89 0.89 0.89;0.853125 0.853125 0.853125;0.81625 0.81625 0.81625;0.779375 0.779375 0.779375;0.7425 0.7425 0.7425;0.705625 0.705625 0.705625;0.66875 0.66875 0.66875;0.631875 0.631875 0.631875;0.595 0.595 0.595;0.558125 0.558125 0.558125;0.52125 0.52125 0.52125;0.484375 0.484375 0.484375;0.4475 0.4475 0.4475;0.410625 0.410625 0.410625;0.37375 0.37375 0.37375;0.336875 0.336875 0.336875;0.3 0.3 0.3;0.28125 0.28125 0.28125;0.2625 0.2625 0.2625;0.24375 0.24375 0.24375;0.225 0.225 0.225;0.20625 0.20625 0.20625;0.1875 0.1875 0.1875;0.16875 0.16875 0.16875;0.15 0.15 0.15],...
    'Color',[1 1 1]);

% Create axes
%axes1 = axes('Parent',figure1,'Color',[1 1 1]);
axes1 = axes('Parent',figure1);
grid on
% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[45 160]);
% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes1,[19 26]);
% Uncomment the following line to preserve the Z-limits of the axes
% zlim(axes1,[-1 1]);
box(axes1,'on');
hold(axes1,'all');

% Create multiple lines using matrix input to plot
% plot1 = plot(X1,YMatrix1,'Parent',axes1,'MarkerEdgeColor',[0 0 0],...
%     'LineWidth',1,...
%     'Color',[0 0 0]);
plot1 = plot(X1,YMatrix1,'Parent',axes1,...
    'LineWidth',1.5);
set(plot1(1),'Marker','diamond','DisplayName',argLegend{1});
set(plot1(2),'Marker','square','LineStyle','-.',...
    'DisplayName',argLegend{2});
set(plot1(3),'Marker','o','LineStyle','-.','DisplayName',argLegend{3});
set(plot1(4),'DisplayName',argLegend{4});

% Create xlabel
xlabel({'K'});

% Create ylabel
ylabel({'implied volatility [%]'});

% Create title
title(titleIn);

% Create legend
legend1 = legend(axes1,'show');
set(legend1,'Color',[1 1 1]);