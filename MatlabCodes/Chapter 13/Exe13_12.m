function H1_HW_COS_vs_MC
clc;clf;close all;
CP  = 'c';  
   
NoOfPaths = 5000;
NoOfSteps = 500;
    
% HW model parameter settings

lambda = 1.12;
eta   = 0.01;
S0    = 100.0;
T     = 15.0;

% We define a ZCB curve (obtained from the market)

P0T = @(T) exp(-0.1*T); 
   
% Strike price range

K = linspace(.01,1.8*S0/P0T(T),20)';
       
% Settings for the COS method

N = 2000;
L = 15;
   
% H1-HW model parameters

gamma = 0.3;
vbar  = 0.05;
v0    = 0.02;
rhoxr = 0.5;
rhoxv =-0.8;
kappa = 0.5;

% Valuation with the COS method

cfH1HW = ChFH1HWModel(P0T,lambda,eta,T,kappa,gamma,vbar,v0,rhoxv, rhoxr);
valCOS = CallPutOptionPriceCOSMthd_StochIR(cfH1HW, CP, S0, T, K, N, L,P0T(T));

% AES simulation of the HHW model

[S_AES,~,Mt_AES] = GeneratePathsHestonHW_AES(NoOfPaths,NoOfSteps,P0T,T,S0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambda,eta);
optValueAES = EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S_AES(:,end),K,Mt_AES(:,end));

% Euler simulation of the HHW model

[S_Euler,~,M_t_Euler] = GeneratePathsHestonHW_Euler(NoOfPaths,NoOfSteps,P0T,T,S0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambda,eta);
optValueEuler = EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S_Euler(:,end),K,M_t_Euler(:,end));

figure(1); hold on;
plot(K,optValueEuler,'-b','linewidth',1.5)
plot(K,optValueAES,'ok','linewidth',1.5)
plot(K,valCOS,'--r','linewidth',1.5)
xlabel('strike,K')
ylabel('Option price')
grid on
legend('Euler','AES','COS')
title('European option prices for the HHW model')

% Exact expectation E(sqrt(v(t)))

function value = meanSqrtV_3(kappa,v0,vbar,gamma_)
delta = 4.0 *kappa*vbar/gamma_/gamma_;
c= @(t) 1.0/(4.0*kappa)*gamma_^2*(1.0-exp(-kappa*(t)));
kappaBar = @(t) 4.0*kappa*v0*exp(-kappa*t)/(gamma_^2*(1.0-exp(-kappa*t)));
value = @(t) sqrt(2.0*c(t))* gamma((1.0+delta)/2.0)/gamma(delta/2.0)*hypergeom(-0.5,delta/2.0,-kappaBar(t)/2.0);

function value = C_H1HW(u,tau,lambd)
i = complex(0.0,1.0);
value = (i*u - 1.0)/lambd * (1-exp(-lambd*tau));

function value = D_H1HW(u,tau,kappa,gamma,rhoxv)
i = complex(0.0,1.0);
D1 = sqrt((kappa-gamma*rhoxv*i*u).^2+(u.^2+i*u)*gamma^2);
g  = (kappa-gamma*rhoxv*i*u-D1)./(kappa-gamma*rhoxv*i*u+D1);
value  = (1.0-exp(-D1*tau))./(gamma^2*(1.0-g.*exp(-D1*tau))).*(kappa-gamma*rhoxv*i*u-D1);
   
function value =A_H1HW(u,tau,P0T,lambd,eta,kappa,gamma,vbar,v0,rhoxv,rhoxr)
i  = complex(0.0,1.0);
D1 = sqrt((kappa-gamma*rhoxv*i*u).^2+(u.*u+i*u)*gamma*gamma);
g  = (kappa-gamma*rhoxv*i*u-D1)./(kappa-gamma*rhoxv*i*u+D1);
   
% Function theta(t)

dt = 0.0001;
f0T = @(t) - (log(P0T(t+dt))-log(P0T(t-dt)))/(2.0*dt);
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));  

% Integration within the function I_1

N  = 500;
z  = linspace(0,tau-1e-10,N);
f1 = (1.0-exp(-lambd*z)).*theta(tau-z);
value1 = trapz(z,f1);
    
% Note that in I_1_adj theta can be time-dependent
% Therefore it is not exactly the same as in the book

I_1_adj = (i*u-1.0) * value1;
I_2     = tau/(gamma^2.0) *(kappa-gamma*rhoxv*i*u-D1) - 2.0/(gamma^2.0)*log((1.0-g.*exp(-D1*tau))./(1.0-g));
I_3     = 1.0/(2.0*power(lambd,3.0))* power(i+u,2.0)*(3.0+exp(-2.0*lambd*tau)-4.0*exp(-lambd*tau)-2.0*lambd*tau);
    
meanSqrtV = meanSqrtV_3(kappa,v0,vbar,gamma);
f2        = meanSqrtV(tau-z).*(1.0-exp(-lambd*z));
value2    = trapz(z,f2);
I_4       = -1.0/lambd * (i*u+u.^2.0)*value2;
value     = I_1_adj + kappa*vbar*I_2 + 0.5*eta^2.0*I_3+eta*rhoxr*I_4;

function cf= ChFH1HWModel(P0T,lambd,eta,tau,kappa,gamma,vbar,v0,rhoxv, rhoxr)
dt = 0.0001;
f0T= @(t) - (log(P0T(t+dt))-log(P0T(t-dt)))/(2.0*dt);
r0 = f0T(0.00001);
C  = @(u) C_H1HW(u,tau,lambd);
D  = @(u) D_H1HW(u,tau,kappa,gamma,rhoxv);
A  = @(u) A_H1HW(u,tau,P0T,lambd,eta,kappa,gamma,vbar,v0,rhoxv,rhoxr);
cf = @(u) exp(A(u) + C(u)*r0 + D(u)*v0);

function [S,time,M_t] = GeneratePathsHestonHW_Euler(NoOfPaths,NoOfSteps,P0T,T,S_0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)

% Time step 

dt = 0.0001;
f0T = @(t) - (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
    
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));

% Generate random noise

Z1 = random('normal',0,1,[NoOfPaths,NoOfSteps]);
Z2 = random('normal',0,1,[NoOfPaths,NoOfSteps]);
Z3 = random('normal',0,1,[NoOfPaths,NoOfSteps]);

% Empty containers for Brownian paths

W1 = zeros([NoOfPaths, NoOfSteps+1]);
W2 = zeros([NoOfPaths, NoOfSteps+1]);
W3 = zeros([NoOfPaths, NoOfSteps+1]);
V = zeros([NoOfPaths, NoOfSteps+1]);
X = zeros([NoOfPaths, NoOfSteps+1]);
R = zeros([NoOfPaths, NoOfSteps+1]);
M_t = ones([NoOfPaths,NoOfSteps+1]);
R(:,1)=r0;
V(:,1)=v0;
X(:,1)=log(S_0);
   
time = zeros([NoOfSteps+1,1]);
        
dt = T / NoOfSteps;
for i = 1:NoOfSteps

    % Making sure that samples from a normal have mean 0 and variance 1

    if NoOfPaths > 1
        Z1(:,i) = (Z1(:,i) - mean(Z1(:,i))) / std(Z1(:,i));
        Z2(:,i) = (Z2(:,i) - mean(Z2(:,i))) / std(Z2(:,i));
        Z3(:,i) = (Z3(:,i) - mean(Z3(:,i))) / std(Z3(:,i));
    end
        W1(:,i+1) = W1(:,i) + power(dt, 0.5)*Z1(:,i);
        W2(:,i+1) = W2(:,i) + power(dt, 0.5)*Z2(:,i);
        W3(:,i+1) = W3(:,i) + power(dt, 0.5)*Z3(:,i);
        
    R(:,i+1) = R(:,i) + lambd*(theta(time(i)) - R(:,i)) * dt + eta* (W1(:,i+1)-W1(:,i));
    M_t(:,i+1) = M_t(:,i) .* exp(0.5*(R(:,i+1) + R(:,i))*dt);
    
    % The variance process

    V(:,i+1) = V(:,i) + kappa*(vbar-V(:,i))*dt+  gamma* sqrt(V(:,i)).* (W2(:,i+1)-W2(:,i));

    % We apply here the truncation scheme to deal with negative values

    V(:,i+1) = max(V(:,i+1),0);
     
    % Simulation of the log-stock price process

    term1 = rhoxr * (W1(:,i+1)-W1(:,i)) + rhoxv * (W2(:,i+1)-W2(:,i))+... 
                + sqrt(1.0-rhoxr^2-rhoxv^2)* (W3(:,i+1)-W3(:,i));
    X(:,i+1) = X(:,i) + (R(:,i) - 0.5*V(:,i))*dt + sqrt(V(:,i)).*term1;       
        
    % Moment matching component, i.e. ensure that E(S(T)/M(T))= S0

    a = S_0 / mean(exp(X(:,i+1))./M_t(:,i+1));
    X(:,i+1) = X(:,i+1) + log(a);
    time(i+1) = time(i) +dt;
end

    % Compute exponent

    S = exp(X);

function [S,time,M_t] = GeneratePathsHestonHW_AES(NoOfPaths,NoOfSteps,P0T,T,S_0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)    

% Time step 

dt = 0.0001;
f0T = @(t) - (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
    
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));
    
% Generate random noise

Z1 = random('normal',0,1,[NoOfPaths,NoOfSteps]);
Z2 = random('normal',0,1,[NoOfPaths,NoOfSteps]);
Z3 = random('normal',0,1,[NoOfPaths,NoOfSteps]);
W1 = zeros([NoOfPaths, NoOfSteps+1]);
W2 = zeros([NoOfPaths, NoOfSteps+1]);
W3 = zeros([NoOfPaths, NoOfSteps+1]);
V = zeros([NoOfPaths, NoOfSteps+1]);
X = zeros([NoOfPaths, NoOfSteps+1]);
R = zeros([NoOfPaths, NoOfSteps+1]);
M_t = ones([NoOfPaths,NoOfSteps+1]);
R(:,1)=r0;
V(:,1)=v0;
X(:,1)=log(S_0);
   
time = zeros([NoOfSteps+1,1]);
        
dt = T / NoOfSteps;
for i = 1:NoOfSteps

    % Making sure that samples from a normal have mean 0 and variance 1

    if NoOfPaths > 1
        Z1(:,i) = (Z1(:,i) - mean(Z1(:,i))) / std(Z1(:,i));
        Z2(:,i) = (Z2(:,i) - mean(Z2(:,i))) / std(Z2(:,i));
        Z3(:,i) = (Z3(:,i) - mean(Z3(:,i))) / std(Z3(:,i));
    end
        W1(:,i+1) = W1(:,i) + power(dt, 0.5)*Z1(:,i);
        W2(:,i+1) = W2(:,i) + power(dt, 0.5)*Z2(:,i);
        W3(:,i+1) = W3(:,i) + power(dt, 0.5)*Z3(:,i);
        
    R(:,i+1) = R(:,i) + lambd*(theta(time(i)) - R(:,i)) * dt + eta* (W1(:,i+1)-W1(:,i));
    M_t(:,i+1) = M_t(:,i) .* exp(0.5*(R(:,i+1) + R(:,i))*dt);
    
    % Exact samples for the variance process

    V(:,i+1) = CIR_Sample(NoOfPaths,kappa,gamma,vbar,0,dt,V(:,i));
         
    k0 = -rhoxv /gamma * kappa*vbar*dt;
    k2 = rhoxv/gamma;
    k1 = kappa*k2 -0.5;
    k3 = sqrt(1.0-rhoxr*rhoxr - rhoxv*rhoxv);
        
    X(:,i+1) = X(:,i) + k0 + (k1*dt - k2)*V(:,i) + R(:,i)*dt + k2*V(:,i+1)+...
                + sqrt(V(:,i)*dt).*(rhoxr*Z1(:,i) + k3 * Z3(:,i));
        
    % Moment matching component, i.e. ensure that E(S(T)/M(T))= S0

    a = S_0 / mean(exp(X(:,i+1))./M_t(:,i+1));
    X(:,i+1) = X(:,i+1) + log(a);
    time(i+1) = time(i) +dt;
end

    % Compute exponent

    S = exp(X);

function optValue = EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S,K,M_t)
optValue = zeros(length(K),1);
if lower(CP) == 'c' || lower(CP) == 1
    for i =1:length(K)
        optValue(i) = mean(1./M_t.*max(S-K(i),0));
    end
elseif lower(CP) == 'p' || lower(CP) == -1
    for i =1:length(K)
        optValue(i) = mean(1./M_t.*max(K(i)-S,0));
    end
end

function sample = CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
sample   = c * random('ncx2',d,kappaBar,[NoOfPaths,1]);

function value = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,tau,K,N,L,P0T)
i = complex(0,1);

% cf   - Characteristic function, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - Interest rate (constant)
% tau  - Time to maturity
% K    - Vector of strike prices
% N    - Number of expansion terms
% L    - Size of truncation domain (typ.:L=8 or L=10)  

x0 = log(S0 ./ K);   

% Truncation domain

a = 0 - L * sqrt(tau); 
b = 0 + L * sqrt(tau);

k = 0:N-1;             % Row vector, index for expansion terms
u = k * pi / (b - a);  % ChF arguments

H_k = CallPutCoefficients('P',a,b,k);
temp = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

% Final output

value = K .* real(mat * temp);

% Use the put-call parity to determine call prices (if needed)

if lower(CP) == 'c' || CP == 1
 value = value + S0 - K * P0T; 
end

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
 psi_k  = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
 psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
 psi_k(1)  = d - c;
 
 chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2); 
 expr1 = cos(k * pi * (d - a)/(b - a)) * exp(d)  - cos(k * pi... 
      * (c - a) / (b - a)) * exp(c);
 expr2 = k * pi / (b - a) .* sin(k * pi * ...
      (d - a) / (b - a))   - k * pi / (b - a) .* sin(k... 
      * pi * (c - a) / (b - a)) * exp(c);
 chi_k = chi_k .* (expr1 + expr2);
 
