function HW_ImpliedVols 
close all; clc;clf;

NoOfPaths = 20000;
NoOfSteps = 1000;

% Market ZCB

P0T       = @(t)exp(-0.1*t);

% Hull-White model parameters

lambda    = 0.02;
eta       = 0.02;

% Call or put option

CP = 'c';

% We compare the ZCB from the market with the Hull-White model

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

% Here we compare the price of an option on a ZCB for Monte Carlo and the analytic expression     

T1 = 4.0;
T2 = 8.0;
K = linspace(0.01, 1.7, 50);

% Monte Carlo paths for the HW model, until time T1

[R,time] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T1,P0T, lambda, eta);

% Discount factor 1/M(T1)

dt = time(2)-time(1);
M_T1 = exp(cumsum(R,2).*dt);
P_T1_T2  = HW_ZCB(lambda,eta,P0T,T1,T2,R(:,end));
OptTheo  = zeros(length(K),1);
OptMC    = zeros(length(K),1);

% Specify call or put option

for i =1:length(K)
    if CP == 'c'
        OptMC(i) = mean(1./M_T1(:,end).*max(P_T1_T2 - K(i),0));
    elseif CP == 'p'
        OptMC(i) = mean(1./M_T1(:,end).*max(K(i) - P_T1_T2,0));
    end
    OptTheo(i) = HW_ZCB_CallPutPrice(CP,K(i),lambda,eta,P0T,T1,T2);
end

figure(2)
plot(K,OptTheo,'-b')
grid on;
hold on
plot(K,OptMC,'.--r')
xlabel('strike, K')
ylabel('Option value')
title('option on a ZCB: mrkt vs. Hull-White')
legend('Theoretical','Monte Carlo')

% Pricing of caplets for different model parameters

frwd = 1.0/(T2-T1) *(P0T(T1)/P0T(T2)-1.0);
K = linspace(frwd/2.0,3.0*frwd,25);
N = 1; % Notional

% Effect of eta

etaV = [0.01, 0.02, 0.03, 0.04];
ivM = zeros(length(K),length(etaV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(etaV)
    etaTemp = etaV(i);
    optPrice  = HW_Caplet_FloorletPrice(CP,N,K,lambda,etaTemp,P0T,T1,T2);
    for j = 1:length(K)    
        optPriceFrwd = optPrice(j)/(T2-T1)/P0T(T2);
        ivM(j,i)  = ImpliedVolatilityBlack76(CP,optPriceFrwd,K(j),T2,frwd,0.3)*100;
    end
    argLegend{idx} = sprintf('eta=%.2f',etaTemp);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'Effect of \eta on implied volatility')

% Effect of lambda

lambdaV = [0.01, 0.03, 0.05, 0.09];
ivM = zeros(length(K),length(etaV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(lambdaV)
    lambdaTemp = lambdaV(i);
    optPrice  = HW_Caplet_FloorletPrice(CP,N,K,lambdaTemp,eta,P0T,T1,T2);
    for j = 1:length(K)    
        optPriceFrwd = optPrice(j)/(T2-T1)/P0T(T2);
        ivM(j,i)  = ImpliedVolatilityBlack76(CP,optPriceFrwd,K(j),T2,frwd,0.3)*100;
    end
    argLegend{idx} = sprintf('lambda=%.2f',lambdaTemp);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'Effect of \lambda on implied volatility')

function impliedVol = ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock,initialVol)
func = @(sigma) (BS_Call_Put_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice).^1.0;
impliedVol = fzero(func,initialVol);
 
% Closed-form expression of European call/put option with Black-Scholes formula

function value=BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1 = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2 = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
 value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
 value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

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

function value = HW_ZCB(lambda,eta,P0T,T1,T2,r0)
A_r = HW_A(lambda,eta,P0T,T1,T2);
B_r = HW_B(lambda,T1,T2);
value = exp(A_r+B_r*r0);

function value = HW_Caplet_FloorletPrice(CP,N,K,lambda,eta,P0T,T1,T2)
N_new = N * (1.0+(T2-T1)*K);
K_new = 1.0 + (T2-T1)*K;
if CP == 'c'
   caplet = N_new.*HW_ZCB_CallPutPrice('p',1.0./K_new,lambda,eta,P0T,T1,T2);
   value= caplet;
elseif CP == 'p'
   floorlet = N_new.*HW_ZCB_CallPutPrice('c',1.0./K_new,lambda,eta,P0T,T1,T2);
   value = floorlet;
end

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

function value = HW_Mu_FrwdMeasure(P0T,lambda,eta,T)
theta = HW_theta(lambda,eta,P0T);
r_0   = HW_r0(P0T);

% Determine moments of process r(t) under the T1-forward measure

thetaFrwd = @(t)(theta(t)+eta^2/lambda^2*(exp(-lambda*(T-t))-1)).*(exp(-lambda*(T-t)));
value     = r_0*exp(-lambda*T)+lambda*integral(thetaFrwd,0,T);

function value = HWVar_r(lambda,eta,T)
value = eta^2/(2*lambda).*(1-exp(-2*lambda*T));

function value = HW_ZCB_CallPutPrice(CP,K,lambda,eta,P0T,T1,T2)

% Coefficients A_r(tau) and B_r(tau)

B=@(T1,T2)HW_B(lambda,T1,T2);
A=@(T1,T2)HW_A(lambda,eta,P0T,T1,T2);

% Strike price adjustments

K_new=K*exp(-A(T1,T2));

% Mean and standard deviation of the HW model under the T-forward measure

muFrwd = HW_Mu_FrwdMeasure(P0T,lambda,eta,T1);
vFrwd = sqrt(HWVar_r(lambda,eta,T1));

% Perform pricing

a  = (log(K_new)-B(T1,T2)*muFrwd)/(B(T1,T2)*vFrwd);
d1 = a-B(T1,T2)*vFrwd;
d2 = a;
term1 = (exp(1/2*B(T1,T2)^2 * vFrwd^2+B(T1,T2)*muFrwd)*normcdf(d1)-K_new.*normcdf(d2));
value = P0T(T1) * exp(A(T1,T2)) * term1 ;

% We use the put-call parity

if CP == 'p'
    value = value - P0T(T2) + K*P0T(T1);
end

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
