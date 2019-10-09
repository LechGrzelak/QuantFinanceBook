function ConvexityCorrection
close all; clc;clf;
format long

NoOfPaths = 20000;
NoOfSteps = 1000;

% Market ZCB

P0T       = @(t)exp(-0.1*t);

% Hull-White model parameters

lambda    = 0.02;
eta       = 0.02;

% We compare the ZCB from the market with the ZCB from the Hull-White model

T_end           = 50;
NumOfGridPoints = 25;
Ti              = linspace(0,T_end,NumOfGridPoints);

% Zero coupon bond from the HW model

r0    = HW_r0(P0T);
Proxy = zeros(NumOfGridPoints,1);
Exact = zeros(NumOfGridPoints,1);
for i= 1: length(Ti)
    Proxy(i) = HW_ZCB(lambda,eta,P0T,0,Ti(i),r0);
    Exact(i) = P0T(Ti(i)); 
end
figure(1);
plot(Ti,Exact,'-b')
grid on;
hold on
plot(Ti,Proxy,'.--r')
xlabel('Maturity, Ti')
ylabel('ZCB, P(0,Ti)')
title('zero-coupon bond: mrkt vs. Hull-White')
legend('Exact','Hull-White')

% Here we define a Libor rate and measure the convexity effect

T1 = 4.0;
T2 = 8.0;

% Monte Carlo paths for the HW model, until time T1

[R,time] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T1,P0T, lambda, eta);
dt = time(2)-time(1);
M_T1 = exp(cumsum(R,2).*dt);
P_T1_T2  = HW_ZCB(lambda,eta,P0T,T1,T2,R(:,end));
L_T1T2   = 1/(T2-T1)*(1./P_T1_T2-1);

% Determine the price from Monte Carlo simulation

ValueMC = mean(1./M_T1(:,end) .* L_T1T2)

% Determine the price using convexity correction

sigma = 0.2;
L0 = 1/(T2-T1)*(P0T(T1)./P0T(T2)-1);
ValueTheoWithoutConvexity = L0
cc = @(sigma)P0T(T2)*(L0 + (T2-T1)*L0^2*exp(sigma.^2*T1))-L0;
ValueTheoWithConvexity = ValueTheoWithoutConvexity + cc(sigma)

% Note that at this point it is unknown which sigma parameter to choose.
% Parameter sigma should be chosen such that the caplet price from the
% Hull- White model and the model for the lognormal Libor yield the same prices
% Below we illustrate the effect of sigma on the convexity adjustment.

figure(2)
sigma_range = 0:0.01:0.6;
plot(sigma_range,cc(sigma_range),'linewidth',2,'color',[0,0.45,0.75]);
grid on;
hold on;
xlabel('$$\sigma$$')
ylabel('$$cc(T_{i-1},T_i)$$')

figure(3)
plot(sigma_range,ValueMC*ones(length(sigma_range),1),'linewidth',2,'color',[0,0.45,0.75]);
ValueWithCC =@(sigma) ValueTheoWithoutConvexity + cc(sigma);
hold on
plot(sigma_range,ValueWithCC(sigma_range),'--r','linewidth',2)
grid on;
xlabel('$$\sigma$$')
ylabel('$$E\left[\frac{\ell(T_{i-1},T_{i-1},T_i)}{M(T_{i-1})}\right]$$')
legend('Market Price','Price with Convexity')

function value = HW_ZCB(lambda,eta,P0T,T1,T2,r0)
A_r = HW_A(lambda,eta,P0T,T1,T2);
B_r = HW_B(lambda,T1,T2);
value = exp(A_r+B_r*r0);

function value = HW_r0(P0T)
bump   = 10e-4;
f_0_t  = @(t)- (log(P0T(t+bump))-log(P0T(t)))/bump;
value  = f_0_t(0.0001);

function value = HW_theta(lambda,eta,P0T)
bump   = 10e-4;
f_0_t  =@(t)- (log(P0T(t+bump))-log(P0T(t)))/bump;
df_dt  =@(t)(f_0_t(t+bump)-f_0_t(t))/bump;
value  =@(t)f_0_t(t)+1/lambda*df_dt(t)+eta^2/(2*lambda^2)*(1-exp(-2*lambda*t));

function value = HW_B(lambda,T1,T2)
value = 1/lambda*(exp(-lambda*(T2-T1))-1);

function value = HW_A(lambda,eta,P0T,T1,T2)
tau   = T2-T1;
B     = @(tau)HW_B(lambda,0,tau);
theta = HW_theta(lambda,eta,P0T);
value = lambda*integral(@(tau)theta(T2-tau).*B(tau),0,tau)+eta^2/(4*lambda^3)...
    * (exp(-2*lambda*tau).*(4*exp(lambda*tau)-1)-3)+eta^2/(2*lambda^2)*tau;

function [R,time] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambda, eta)

% Time step 

dt = 0.0001;

% Complex number

f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
   
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);  
theta = @(t) 1.0/lambda * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambda*lambda)*(1.0-exp(-2.0*lambda*t));  

% Define initial value

R=zeros(NoOfPaths,NoOfSteps);
R(:,1) = r0;

% Random noise

Z=random('normal',0,1,[NoOfPaths,NoOfSteps]);
W=zeros([NoOfPaths,NoOfSteps]);

dt = T / NoOfSteps;
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps
    if NoOfPaths>1
        Z(:,i)   = (Z(:,i) - mean(Z(:,i))) / std(Z(:,i));
    end
    W(:,i+1)  = W(:,i) + sqrt(dt).*Z(:,i);
    R(:,i+1)  = R(:,i) + lambda*(theta(time(i))-R(:,i))*dt+  eta* (W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end
