function BS_Hedging
clf;clc;close all;
format long

% Model configuration

S0         = 1;
NoOfPaths = 1000;
NoOfSteps = 1000;
r         = 0.1;
sigma     = 0.2;
T         = 1;
K         = 1*0.95;

% Generation of the stock paths

randn('seed',11) %17 %20 %25 %43 %4 %11
[S,time] = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0);

% Define lambda expression for call option and for delta

C = @(t,K,S0)BS_Call_Put_Option_Price('c',S0,K,sigma,t,T,r);
Delta_call = @(t,K,S0)BS_Delta('c',S0,K,sigma,t,T,r);

% Hedging part

PnL        = zeros(NoOfPaths,length(time));
delta_init = Delta_call(0,K,S0);
PnL(:,1)   = C(0,K,S0) - delta_init*S0;

CallM      = zeros(NoOfPaths,length(time));
CallM(:,1) = C(0,K,S0);
DeltaM     = zeros(NoOfPaths,length(time));
DeltaM(:,1)= Delta_call(0,K,S0);

% Iterate over all the time steps

for i=2:NoOfSteps+1
    time(i)
    dt = time(i)-time(i-1);   
    delta_old  = Delta_call(time(i-1),K,S(:,i-1));
    delta_curr = Delta_call(time(i),K,S(:,i));
    PnL(:,i)   =  PnL(:,i-1)*exp(r*dt) - (delta_curr-delta_old).*S(:,i); % PnL
    CallM(:,i) = C(time(i),K,S(:,i));
    DeltaM(:,i)= delta_curr;
end

% Final PnL calculation 

PnL(:,end)=PnL(:,end) -max(S(:,end)-K,0) +  DeltaM(:,end).*S(:,end);

% Plot hedging results for a specified path

path_id =21;
 
figure(1)
plot(time,S(path_id,:),'linewidth',1.5)
hold on
plot(time,CallM(path_id,:),'-m','linewidth',1.5)
plot(time,PnL(path_id,:),'-r','linewidth',1.5)
plot(time,DeltaM(path_id,:),'g','linewidth',1.5)
plot(time,K.*ones(length(time),1),'--k','linewidth',1.5)
legend('Stock value','Call price','P&L','\Delta')
grid on
grid minor
title('Hedging of a Call opton')
xlabel('time')
xlim([0,T+0.01])

% PnL histogram

figure(2)
hist(PnL(:,end),50)
xlim([-0.1,0.1])
grid on
title('PnL Histogram with respect to number of re-hedging')

% Analysis for each path

for i =1:NoOfPaths
    fprintf('path_id = %2.0d, PnL(t_0)= %0.4f, PnL(Tm-1) =%0.4f,S(Tm) =%0.4f,max(S(tm)-K,0)= %0.4f, PnL(t_m) = %0.4f',i,PnL(i,1),PnL(i,end-1),S(i,end),max(S(i,end)-K,0),PnL(i,end))
    fprintf('\n')
end

function [S,time] = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0)

% Approximation

S=zeros(NoOfPaths,NoOfSteps);
S(:,1) = S0;

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
    S(:,i+1) = S(:,i) + r * S(:,i)*dt + sigma * S(:,i).*(W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end

function value = BS_Delta(CP,S_0,K,sigma,t,T,r)
if (t-T>10e-20 && T-t<10e-7)
   t=T;
end
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
if lower(CP) == 'c' || lower(CP) == 1
   value = normcdf(d1);
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(d1)-1;
end

% Black-Scholes call option price

function value = BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r)
if (t-T>10e-20 && T-t<10e-7)
   t=T;
end
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
d2    = d1 - sigma * sqrt(T-t);
if lower(CP) == 'c' || lower(CP) == 1
    value = normcdf(d1) .* S_0 - normcdf(d2) .* K * exp(-r * (T-t));
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(-d2) .* K*exp(-r*(T-t)) - normcdf(-d1).*S_0;
end
