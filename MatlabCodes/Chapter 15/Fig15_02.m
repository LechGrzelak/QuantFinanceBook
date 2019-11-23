function Calibration_H1HW_FX
close all;clc;

% Option type

CP  = 'c';

% Specify the maturity time for the calibration

index = 2;
    
T_market  = [0.5,1.0,5.0,10.0,20.0,30.0];
IV_Market = [[11.41 , 10.49 , 9.66 , 9.02 , 8.72 , 8.66 , 8.68 ];
             [12.23 , 10.98 , 9.82 , 8.95 , 8.59 , 8.59 , 8.65];
             [13.44 , 11.84 , 10.38 , 9.27 , 8.76 , 8.71 , 8.83];
             [16.43 , 14.79 , 13.34 , 12.18 , 11.43 , 11.07 , 10.99];
             [22.96 , 21.19 , 19.68 , 18.44 , 17.50 , 16.84 , 16.46];
             [25.09 , 23.48 , 22.17 , 21.13 , 20.35 , 19.81 , 19.48]];
T = T_market(index);
referenceIV = IV_Market(index,:)/100.0;
    
% Settings for the COS method

N = 500;
L = 8;  
    
% ZCB from the domestic and foreign markets

P0Td = @(t) exp(-0.02*t);
P0Tf = @(t) exp(-0.05*t);
    
y0      = 1.35;
frwdFX  = y0*P0Tf(T)/P0Td(T);

% Fixed mean reversion parameter

kappa = 0.5;

% HW model parameters

lambdad  = 0.001;
lambdaf  = 0.05;
etad    = 0.007;
etaf    = 0.012;
    
% Correlations

Rxrd  = -0.15;
Rxrf  = -0.15;
Rvrd  = 0.3;
Rvrf  = 0.3;
Rrdrf = 0.25;
    
% Strike and option prices from market implied volatilities

K = GenerateStrikes(frwdFX,T)';
referencePrice = P0Td(T)* BS_Call_Put_Option_Price(CP,frwdFX,K,referenceIV',T,0.0);
       
% Calibrate the H1HW model and show the output

figure(1)
title('Calibration of the H1HW model')
plot(K,referencePrice)
xlabel('strike, K')
ylabel('reference option price')
grid on
hold on
  
% Global search algorithm
% The search will stop after running 30 seconds 

gs = GlobalSearch('FunctionTolerance',1e-3,'NumTrialPoints',300,'XTolerance',1e-3,'MaxTime',30);

% The HHWFX model calibration

% x= [gamma,vBar,Rxv,v0]
trgt = @(x)TargetVal_HHWFX(CP,referencePrice,P0Td,T,K,frwdFX,x(1),x(2),x(3),x(4),Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdad,etad,lambdaf,etaf,kappa);
%TargetVal_HHWFX(CP,kappa,x(1),x(2),Rxr,x(3),x(4),x(5),eta,lambda,K,marketPrice,S0,T,P0T);

% The bounds and initial values

x0 = [1.0, 0.005,-0.7, 0.004];
lowerBound = [0.1, 0.001,-0.99, 0.0001];
upperBound = [0.8,  0.4,  -0.3, 0.4];

problemHHWFX = createOptimProblem('fmincon','x0',x0,'objective',trgt,'lb',lowerBound,'ub',upperBound);
x_HHWFX = run(gs,problemHHWFX);

% Optimal set of parameters

gamma_HHWFX_opt = x_HHWFX(1);
vBar_HHWFX_opt  = x_HHWFX(2);
Rxv_HHWFX_opt   = x_HHWFX(3);
v0_HHWFX_opt   = x_HHWFX(4);

% Plot final result

cfHHW_FX = @(u)ChFH1HW_FX(u,T,gamma_HHWFX_opt,Rxv_HHWFX_opt,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdad,etad,lambdaf,etaf,kappa,vBar_HHWFX_opt,v0_HHWFX_opt);
valCOS_HHWFX = P0Td(T)* CallPutOptionPriceCOSMthd_StochIR(cfHHW_FX, CP, frwdFX, T, K, N, L,1.0);
plot(K,valCOS_HHWFX,'--r')
legend('mrkt opt price','HHW-FX opt Price')
xlabel('strike')
ylabel('option price')
fprintf('Optimal parameters HHW-FX are: gamma = %.4f, vBar = %.4f, Rrv = %.4f, v0 = %.4f', gamma_HHWFX_opt,vBar_HHWFX_opt,Rxv_HHWFX_opt,v0_HHWFX_opt);
fprintf('\n')

% Implied volatilities, the COS method vs. closed-form expression

IVHHWFX = zeros(length(K),1);
IVMrkt = zeros(length(K),1);
for i=1:length(K)
    valCOS_HHWFX_frwd = valCOS_HHWFX(i) / P0Td(T);
    valMrktFrwd = referencePrice(i)/P0Td(T);
    IVHHWFX(i) = ImpliedVolatilityBlack76(CP,valCOS_HHWFX_frwd,K(i),T,frwdFX,0.3)*100.0;
    IVMrkt(i) = ImpliedVolatilityBlack76(CP,valMrktFrwd,K(i),T,frwdFX,0.3)*100.0;
end

figure(2)
plot(K,IVMrkt,'--r');hold on
plot(K,IVHHWFX,'ok')
legend('IV-Mrkt','IV-HHW-FX')
grid on


function value = GenerateStrikes(frwd,Ti)
c_n = [-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5];
value =  frwd * exp(0.1 * c_n * sqrt(Ti));

function value = TargetVal_HHWFX(CP,marketPrice,P0Td,T,K,frwdFX,gamma,vBar,Rxv,v0,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdad,etad,lambdaf,etaf,kappa)

% Settings for the COS method

N = 1000;
L = 15;
cfHHW_FX = @(u)ChFH1HW_FX(u,T,gamma,Rxv,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdad,etad,lambdaf,etaf,kappa,vBar,v0);
valCOS   = P0Td(T)* CallPutOptionPriceCOSMthd_StochIR(cfHHW_FX, CP, frwdFX, T, K, N, L,1.0);

% Error is a squared sum

value = sum((marketPrice-valCOS).^2);
fprintf('HHWFX- Total Error = %.4f',value)
fprintf('\n')

function value = C_H1HW_FX(u,tau,kappa,gamma,Rxv)
i = complex(0.0,1.0);
D1 = sqrt((kappa-gamma*Rxv*i*u).^2.0+(u.^2+i*u)*gamma^2);
g  = (kappa-gamma*Rxv*i*u-D1)./(kappa-gamma*Rxv*i*u+D1);
value  = (1.0-exp(-D1.*tau))./(gamma^2*(1.0-g.*exp(-D1.*tau))).*(kappa-gamma*Rxv*i*u-D1);

function cf= ChFH1HW_FX(u,tau,gamma,Rxv,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa,vBar,v0)
i  = complex(0.0,1.0);
C  = @(u,tau) C_H1HW_FX(u,tau,kappa,gamma,Rxv);
Bd = @(t,T) 1.0/lambdd*(exp(-lambdd*(T-t))-1.0);
Bf = @(t,T) 1.0/lambdf*(exp(-lambdf*(T-t))-1.0);
G  = meanSqrtV_3(kappa,v0,vBar,gamma);
    
zeta = @(t) (Rxrd*etad*Bd(t,tau) - Rxrf*etaf*Bf(t,tau)).*G(t) +...
    Rrdrf*etad*etaf*Bd(t,tau).*Bf(t,tau) - 0.5*(etad^2*Bd(t,tau).^2+etaf^2*Bf(t,tau).^2);
    
% Integration within the function A(u,tau)

N    = round(25*tau);
z    = linspace(0.0+1e-10,tau-1e-10,N)';
temp1 =@(z1) kappa*vBar + Rvrd*gamma*etad*G(tau-z1).*Bd(tau-z1,tau);
temp2 =@(z1,u) -Rvrd*gamma*etad*G(tau-z1).*Bd(tau-z1,tau)*i*u;
temp3 =@(z1,u)  Rvrf*gamma*etaf*G(tau-z1).*Bf(tau-z1,tau)*i*u;
f = @(z1,u) (temp1(z1)+temp2(z1,u)+temp3(z1,u)).*C(u,z1);
int1=trapz(z,f(z,u)) ;
 
% int1= zeros(1,length(u));
% for k = 1:length(u)
%     temp1   = kappa*vBar + Rvrd*gamma*etad*G(tau-z).*Bd(tau-z,tau);
%     temp2   = -Rvrd*gamma*etad*G(tau-z).*Bd(tau-z,tau)*i*u(k);
%     temp3   = Rvrf*gamma*etaf*G(tau-z).*Bf(tau-z,tau)*i*u(k);
%     f       = (temp1+temp2+temp3).*C(u(k),z);
%     int1(k) = trapz(z,f);
% end

int2 = (u.^2 + i*u)*trapz(z,zeta(tau-z));
    
A = int1 + int2;
cf = exp(A + v0*C(u,tau));


% Exact expectation E(sqrt(v(t)))

function value = meanSqrtV_3(kappa,v0,vbar,gamma_)
delta = 4.0 *kappa*vbar/gamma_/gamma_;
c= @(t) 1.0/(4.0*kappa)*gamma_^2*(1.0-exp(-kappa*(t)));
kappaBar = @(t) 4.0*kappa*v0*exp(-kappa*t)./(gamma_^2*(1.0-exp(-kappa*t)));
value = @(t) sqrt(2.0*c(t)).*gamma((1.0+delta)/2.0)/gamma(delta/2.0).*hypergeom(-0.5,delta/2.0,-kappaBar(t)/2.0);

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
