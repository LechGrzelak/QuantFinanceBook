function Calibration_SZHW_and_H1HW

% The SZHW and the H1-HW models where calibration takes place for 
% a single maturity time. 

clc;clf;close all;
CP  = 'c';  

% HW model parameter settings

lambda = 0.01;
eta   = 0.01;
S0    = 1145.88;       
   
% Fixed SZHW / H1-HW model parameters

kappa =  0.5;
Rxr   =  0.3;
      
% We define a ZCB curve (obtained from the market)
% This curve is based on an estimation from real market data

P0T = @(T) exp(0.0012*T+0.0007);
     
% Settings for the COS method

N = 2000;
L = 15;
    
%################## Here we define market option prices #################
    
T = 0.5;
referenceIV  = [57.61, 22.95, 15.9]'/100.0;
atmVol = referenceIV(2);
    
%T = 1.0;
%referenceIV = [48.53, 24.49, 19.23]'/100.0;
%atmVol = referenceIV(2);
        
%T = 10.0;
%referenceIV = [36.76, 29.18, 27.66]'/100.0;
%atmVol = referenceIV(2);
    
% Strike price range

frwd = S0/P0T(T);
K = [0.4*frwd, frwd, 1.2*frwd]';
   
marketPrice = P0T(T)* BS_Call_Put_Option_Price(CP,S0 / P0T(T),K,referenceIV,T,0.0);
   
figure(1); hold on;
plot(K,marketPrice,'-r')
xlabel('strike,K')
ylabel('Optio price')
grid on

% Global search algorithm
% The search will terminate after running 30 seconds

gs = GlobalSearch('FunctionTolerance',1e-3,'NumTrialPoints',300,'XTolerance',1e-3,'MaxTime',30);

% The SZHW model calibration

%x = [gamma,sigmaBar,Rrsigma,Rxsigma,sigma0]
trgt = @(x)TargetVal_SZHW(CP,kappa,x(1),x(2),Rxr,x(3),x(4),x(5),eta,lambda,K,marketPrice,S0,T,P0T);
lowerBound = [0.1, 0.01,-0.85, -0.85, 0.01];
upperBound = [0.8,  0.4, 0.85, -0.0, 0.8];
x0  = [0.1, atmVol, 0.0,-0.5,atmVol];
problemSZHW = createOptimProblem('fmincon','x0',x0,'objective',trgt,'lb',lowerBound,'ub',upperBound);
x_SZHW = run(gs,problemSZHW);

% Optimal set of parameters

gamma_SZHW_opt    = x_SZHW(1);
sigmaBar_SZHW_opt = x_SZHW(2);
Rrsigma_SZHW_opt  = x_SZHW(3);
Rxsigma_SZHW_opt  = x_SZHW(4);
sigma0_SZHW_opt   = x_SZHW(5);

% The H1-HW model calibration

% x = [gamma,vBar,Rxv,v0]
trgt = @(x)TargetVal_H1HW(CP,kappa,x(1),x(2),Rxr,x(3),x(4),eta,lambda,K,marketPrice,S0,T,P0T);
lowerBound = [0.1, 0.001,-0.95, 0.01];
upperBound = [1.1,  0.6,  -0.2, 0.4];
% [gamma,vBar,Rxv,v0]
x0= [1.0, atmVol*atmVol,-0.7, atmVol*atmVol];
problemH1HW = createOptimProblem('fmincon','x0',x0,'objective',trgt,'lb',lowerBound,'ub',upperBound);
x_H1HW = run(gs,problemH1HW);

% Optimal set of parameters

gamma_H1_opt    = x_H1HW(1);
vBar_H1_opt     = x_H1HW(2);
Rxv_H1_opt      = x_H1HW(3);
v0_H1_opt       = x_H1HW(4);

% Plot final result

cf_SZHW = @(u)ChFSZHW(u,P0T,T,kappa,sigmaBar_SZHW_opt,gamma_SZHW_opt,lambda,eta,Rxsigma_SZHW_opt,Rxr,Rrsigma_SZHW_opt,sigma0_SZHW_opt);
valCOS_SZHW    = CallPutOptionPriceCOSMthd_StochIR(cf_SZHW,CP,S0,T,K,N,L,P0T(T));
plot(K,valCOS_SZHW,'-b')
cfH1HW = ChFH1HWModel(P0T,lambda,eta,T,kappa,gamma_H1_opt,vBar_H1_opt,v0_H1_opt,Rxv_H1_opt, Rxr);
valCOS_H1HW = CallPutOptionPriceCOSMthd_StochIR(cfH1HW, CP, S0, T, K, N, L,P0T(T));
plot(K,valCOS_H1HW,'ok')
legend('mrkt opt price','SZHW opt price','H1-HW opt Price')
xlabel('strike')
ylabel('option price')
fprintf('Optimal parameters SZHW are: gamma = %.4f, sigmaBar = %.4f, Rrsigma = %.4f, Rxsigma = %.4f, sigma0 = %.4f ', x_SZHW(1),x_SZHW(2),x_SZHW(3),x_SZHW(4),x_SZHW(5))
fprintf('\n')
fprintf('Optimal parameters H1-HW are: gamma = %.4f, vBar = %.4f, Rrv = %.4f, v0 = %.4f', x_H1HW(1),x_H1HW(2),x_H1HW(3),x_H1HW(4))
fprintf('\n')

% Implied volatilities, the COS method vs. closed-form solution

IVSZHW = zeros(length(K),1);
IVH1HW = zeros(length(K),1);
IVMrkt = zeros(length(K),1);
for i=1:length(K)
    frwdStock = S0 / P0T(T);
    valCOS_SZHWFrwd = valCOS_SZHW(i) / P0T(T);
    valCOS_H1HWFrwd = valCOS_H1HW(i) / P0T(T);
    valMrktFrwd = marketPrice(i)/P0T(T);
    IVSZHW(i) = ImpliedVolatilityBlack76(CP,valCOS_SZHWFrwd,K(i),T,frwdStock,0.3)*100.0;
    IVH1HW(i) = ImpliedVolatilityBlack76(CP,valCOS_H1HWFrwd,K(i),T,frwdStock,0.3)*100.0;
    IVMrkt(i) = ImpliedVolatilityBlack76(CP,valMrktFrwd,K(i),T,frwdStock,0.3)*100.0;
end

figure(2)
plot(K,IVMrkt,'--r');hold on
plot(K,IVSZHW,'-b')
plot(K,IVH1HW,'ok')
legend('IV-Mrkt','IV-H1HW','IV-SZHW')
grid on

IVMrkt-IVSZHW
IVMrkt-IVH1HW

stop=1

function value = TargetVal_SZHW(CP,kappa,gamma,sigmaBar,Rxr,Rrsigma,Rxsigma,sigma0,eta,lambda,K,marketPrice,S0,T,P0T)

% Settings for the COS method

N = 1000;
L = 15;
cf = @(u)ChFSZHW(u,P0T,T,kappa, sigmaBar,gamma,lambda,eta,Rxsigma,Rxr,Rrsigma,sigma0);
valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));

% Error is a squared sum

value = sum((marketPrice-valCOS).^2);
fprintf('SZHW- Total Error = %.4f',value)
fprintf('\n')

function value = TargetVal_H1HW(CP,kappa,gamma,vBar,Rxr,Rxv,v0,eta,lambda,K,marketPrice,S0,T,P0T)

% Settings for the COS method

N = 1000;
L = 15;

% Valuation with the COS method

cfH1HW = ChFH1HWModel(P0T,lambda,eta,T,kappa,gamma,vBar,v0,Rxv, Rxr);
valCOS = CallPutOptionPriceCOSMthd_StochIR(cfH1HW, CP, S0, T, K, N, L,P0T(T));

% Error is a squared sum

value = sum((marketPrice-valCOS).^2);
fprintf('H1-HW- Total Error = %.4f',value)
fprintf('\n')

function value = ChFSZHW(u,P0T,tau,kappa, sigmabar,gamma,lambda,eta,Rxsigma,Rxr,Rrsigma,sigma0)
v_D = D(u,tau,kappa,Rxsigma,gamma);
v_E = E(u,tau,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar);
v_A = A(u,tau,P0T,eta,lambda,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar);

% Initial variance 

v0     =sigma0^2;

% Characteristic function of the SZHW model

value = exp(v0*v_D + sigma0*v_E + v_A);
        
function value=C(u,tau,lambda)
    i     = complex(0,1);
    value = 1/lambda*(i*u-1).*(1-exp(-lambda*tau)); 

function value=D(u,tau,kappa,Rxsigma,gamma)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d=sqrt(a_1.^2-4*a_0.*a_2);
    g=(-a_1-d)./(-a_1+d);    
value=(-a_1-d)./(2*a_2*(1-g.*exp(-d*tau))).*(1-exp(-d*tau));

function value=E(u,tau,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d  =sqrt(a_1.^2-4*a_0.*a_2);
    g =(-a_1-d)./(-a_1+d);    
    
    c_1=gamma*Rxsigma*i*u-kappa-1/2*(a_1+d);
    f_1=1./c_1.*(1-exp(-c_1*tau))+1./(c_1+d).*(exp(-(c_1+d)*tau)-1);
    f_2=1./c_1.*(1-exp(-c_1*tau))+1./(c_1+lambda).*(exp(-(c_1+lambda)*tau)-1);
    f_3=(exp(-(c_1+d)*tau)-1)./(c_1+d)+(1-exp(-(c_1+d+lambda)*tau))./(c_1+d+lambda);
    f_4=1./c_1-1./(c_1+d)-1./(c_1+lambda)+1./(c_1+d+lambda);
    f_5=exp(-(c_1+d+lambda).*tau).*(exp(lambda*tau).*(1./(c_1+d)-exp(d*tau)./c_1)+exp(d*tau)./(c_1+lambda)-1./(c_1+d+lambda)); 

    I_1=kappa*sigmabar./a_2.*(-a_1-d).*f_1;
    I_2=eta*Rxr*i*u.*(i*u-1)./lambda.*(f_2+g.*f_3);
    I_3=-Rrsigma*eta*gamma./(lambda.*a_2).*(a_1+d).*(i*u-1).*(f_4+f_5);
    value=exp(c_1*tau).*1./(1-g.*exp(-d*tau)).*(I_1+I_2+I_3);
    
function value=A(u,tau,P0T,eta,lambda,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d  =sqrt(a_1.^2-4*a_0.*a_2);
    g =(-a_1-d)./(-a_1+d); 
    f_6=eta^2/(4*lambda^3)*(i+u).^2*(3+exp(-2*lambda*tau)-4*exp(-lambda*tau)-2*lambda*tau);
    A_1=1/4*((-a_1-d)*tau-2*log((1-g.*exp(-d*tau))./(1-g)))+f_6;
   
    % Integration within the function A(u,tau)

    value=zeros(1,length(u));   
    N=500;
    arg=linspace(0,tau,N);

    % Solve the integral for A

    for k=1:length(u)
       E_val=E(u(k),arg,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar);
       C_val=C(u(k),arg,lambda);
       f=(kappa*sigmabar+1/2*gamma^2*E_val+gamma*eta*Rrsigma*C_val).*E_val;
       value(1,k)=trapz(arg,f);
    end
    value = value + A_1;   
    
% Correction for the interest rate term structure    

help = eta^2/(2*lambda^2)*(tau+2/lambda*(exp(-lambda*tau)-1)-1/(2*lambda)*(exp(-2*lambda*tau)-1));
correction = (i*u-1).*(log(1/P0T(tau))+help);
value = value + correction;

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
% Therefore it is not exactly as in the book

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
temp(1) = 0.5 * temp(1);     % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u); % Matrix-vector manipulations

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
 
 % Closed-form expression of European call/put option with Black-Scholes formula

function value=BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * tau) ./ (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function impliedVol = ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock,initialVol)
func = @(sigma) (BS_Call_Put_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice).^1.0;
impliedVol = fzero(func,initialVol);
