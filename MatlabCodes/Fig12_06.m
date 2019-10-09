function Exposures_HW_Netting
close all; clc;clf;
NoOfPaths = 2000;
NoOfSteps = 1000;
CP        = 'payer';
notional  = 10000.0; 
notional2 = 10000.0;
alpha     = 0.95;
alpha2    = 0.99;

% Market ZCB

P0T       = @(t)exp(-0.01*t);

% Hull-White model parameters

lambda     = 0.5;
eta        = 0.03;

% We compare ZCB from the market with the Hull-White model

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

% Here we simulate the exposure profiles for a swap, using the HW model    

% Swap settings

K = 0.01;  % Strike
Ti = 1.0;  % Beginning of the swap
Tm = 10.0; % End date of the swap 
n = 10;    % Number of payments between Ti and Tm

% Monte Carlo paths

randn('seed',1)
[R,timeGrid,M_t] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,Tm+0.1,P0T,lambda, eta);

% Compute value of the porftolio, E, EE and PFE

Value = zeros([NoOfPaths,NoOfSteps+1]);
E     = zeros([NoOfPaths,NoOfSteps+1]);
EE    = zeros([NoOfSteps+1,1]);
NEE    = zeros([NoOfSteps+1,1]);
PFE   = zeros([NoOfSteps+1,1]);
PFE2   = zeros([NoOfSteps+1,1]);

for idx = 1:length(timeGrid)
    V = HW_SwapPrice(CP,notional,K,timeGrid(idx),Ti,Tm,n,R(:,idx),P0T,lambda,eta);
    Value(:,idx) = V;
    E(:,idx)     = max(V,0.0);
    EE(idx)      = mean(E(:,idx)./M_t(:,idx));
    NEE(idx)      = mean(max(-V,0.0)./M_t(:,idx));
    PFE(idx)     = quantile(E(:,idx),alpha);
    PFE2(idx)     = quantile(E(:,idx),alpha2);
end


% Compute value of the porftolio, E, EE and PFE

ValuePortf = zeros([NoOfPaths,NoOfSteps+1]);
EPort     = zeros([NoOfPaths,NoOfSteps+1]);
EEPort    = zeros([NoOfSteps+1,1]);
PFEPort   = zeros([NoOfSteps+1,1]);

for idx = 1:length(timeGrid)
    Swap1 = HW_SwapPrice(CP,notional,K,timeGrid(idx),Ti,Tm,n,R(:,idx),P0T,lambda,eta);
    Swap2 = HW_SwapPrice(CP,notional2,0.0,timeGrid(idx),Tm-2.0*(Tm-Ti)/n,Tm,1,R(:,idx),P0T,lambda,eta);
    VPort = (Swap1-Swap2);
    ValuePortf(:,idx)=VPort;
    EPort(:,idx)     = max(VPort,0.0);
    EEPort(idx)      = mean(EPort(:,idx)./M_t(:,idx));
    PFEPort(idx)     = quantile(EPort(:,idx),alpha);
end

figure(2)
plot(timeGrid,Value(1:15,:),'color',[0,0.45,0.75],'linewidth',1.5)
%plot(timeGrid,Value(1:25,:),'linewidth',1.5)
grid()
xlabel('time')
ylabel('exposure, Value(t)')
title('Paths of the price V^S(t)')
a= axis;

figure(3)
plot(timeGrid,E(1:15,:),'color',[0,0.45,0.75],'linewidth',1.5)
grid()
xlabel('time')
ylabel('exposure, E(t)')
title('Paths of positive exposure E(t)=max(V^S(t),0)')
axis(a)
    
figure(4)
plot(timeGrid,EE,'r','linewidth',1.5)
grid on
xlabel('time')
ylabel('exposure, EE(t)')
title('Discounted Expected (positive) exposure, EE')
    
figure(5)
plot(timeGrid,EE,'r','linewidth',1.5)
hold on
plot(timeGrid,PFE,'k','linewidth',1.5)
grid on
xlabel('time')
ylabel('exposure, EE(t)')
legend('EE','PFE')
title('Exposure profiles, EE and PFE (single swap)')
a=axis;

figure(6)
plot(timeGrid,EEPort,'r','linewidth',1.5)
hold on
plot(timeGrid,PFEPort,'k','linewidth',1.5)
grid on
xlabel('time')
ylabel('exposure, EE(t)')
legend('EE-portfolio','PFE-portfolio')
title('Exposure profiles, EE and PFE (porftolio)')
axis(a);

figure(7)
hold on
plot(timeGrid,EE,'r','linewidth',1.5)
plot(timeGrid,EEPort,'--r','linewidth',1.5)
grid()
title('Comparison of EEs ')
legend('EE, single swap','EE, portfolio')

figure(8)
hold on
plot(timeGrid,PFE,'k','linewidth',1.5)
plot(timeGrid,PFEPort,'--k','linewidth',1.5)
grid()
title('Comparison of PFEs ')
legend('PFE, single swap','PFE, portfolio')

figure(9)
plot(timeGrid,EE,'r','linewidth',1.5)
hold on
plot(timeGrid,NEE,'--k','linewidth',1.5)
grid on
xlabel('time')
ylabel('exposure')
legend('EE','NEE')
title('Expected positve and negative exposures, EE & NEE')

figure(10)
plot(timeGrid,EE,'r','linewidth',1.5)
hold on
plot(timeGrid,PFE,'k','linewidth',1.5)
plot(timeGrid,PFE2,'--b','linewidth',1.5)
grid on
xlabel('time')
ylabel('exposure, EE(t)')
legend('EE','PFE at level 0.95','PFE at level 0.99')
title('Exposure profiles, EE and PFE (single swap)')
a=axis;

function [R,time,M_t] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambda, eta)

% Time step 

dt = 0.0001;

% Complex number

f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
   
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);  
theta = @(t) 1.0/lambda * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambda*lambda)*(1.0-exp(-2.0*lambda*t));  

% Define initial value

R  = zeros(NoOfPaths,NoOfSteps);
R(:,1) = r0;
M_t = ones(NoOfPaths,NoOfSteps);

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
    M_t(:,i+1) = M_t(:,i) .* exp(0.5*(R(:,i+1) + R(:,i))*dt);
    time(i+1) = time(i) + dt;
end

function value = HW_ZCB(lambda,eta,P0T,T1,T2,r0)
if T1>T2
    value = ones([length(r0),1]);
else
    A_r = HW_A(lambda,eta,P0T,T1,T2);
    B_r = HW_B(lambda,T1,T2);
    value = exp(A_r+B_r*r0);
end

function swap = HW_SwapPrice(CP,notional,K,t,Ti,Tm,n,r_t,P0T,lambda,eta)

% CP  - Payer of receiver
% n   - Notional
% K   - Strike price
% t   - Today's date
% Ti  - Beginning of the swap
% Tm  - End of Swap
% n   - Number of dates payments between Ti and Tm
% r_t - Interest rate at time t

% Equally spaced grid of payments

ti_grid = linspace(Ti,Tm,n);
if length(ti_grid)==1
    ti_grid = [Ti,Tm];
end
tau = ti_grid(2)- ti_grid(1);
P_t_Ti = @(T) HW_ZCB(lambda,eta,P0T,t,T,r_t);

swap = zeros([length(r_t),1]);
%floatLeg = zeros([length(r_t),1]);
%fixedLeg = zeros([length(r_t),1]);

for idx = 2:length(ti_grid) %Tk = ti_grid(2:end)
    Tkprev = ti_grid(idx-1);
    Tk = ti_grid(idx);
    if t<Tk
        L = 1/tau*(P_t_Ti(Tkprev)./P_t_Ti(Tk)-1);
        if CP=='payer'
            swap = swap + tau*(P_t_Ti(Tk).*(L-K)); 
            %floatLeg = floatLeg + tau*P_t_Ti(Tk).*L;
            %fixedLeg = fixedLeg + tau*(P_t_Ti(Tk).*K);
        elseif CP == 'receiver'
            swap = swap + tau*(P_t_Ti(Tk).*(K-L));
        end
    end
end
swap = swap *notional;

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

function value = HW_r0(P0T)
bump   = 10e-4;
f_0_t  = @(t)- (log(P0T(t+bump))-log(P0T(t)))/bump;
value  = f_0_t(0.0001);
