function BSHW_TermStructure
clc;clf;close all;

% Characteristic function parameters 
% HW model parameter settings

lambda = 0.1;
eta   = 0.01;
sigma = 0.2;
rho   = 0.3;
S0    = 100; 

% Strike equals stock value, thus ATM

K  = [100];
CP = 'c';

% We define a ZCB curve (obtained from the market)

P0T = @(T) exp(-0.05*T); 
 
% Settings for the COS method

N = 100;
L = 8;

% Range of maturities for the BSHW model

TMat = linspace(0.1,5.0,20);

% Effect of lambda

lambdaV   = [0.001,0.1,0.5,1.5];
argLegend = cell(4,1);
idy = 1;
IV = zeros(length(TMat),length(lambdaV));
for i=1:length(lambdaV)
    lambdaTemp = lambdaV(i); 
    for idx = 1:1:length(TMat)
        T= TMat(idx);
        cf        = @(u) ChFBSHW(u, T, P0T, lambdaTemp, eta, rho, sigma);
        valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
        frwdStock = S0 / P0T(T);
        valCOSFrwd = valCOS / P0T(T);
        IV(idx,idy) = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock,0.3);
    end
    argLegend{i} = sprintf('lambda=%.3f',lambdaTemp);
    idy = idy +1;
end
MakeFigure(TMat, IV*100,argLegend,'Implied ATM volatility for parameter \lambda')

% Effect of eta

etaV   = [0.001,0.05,0.1,0.15];
argLegend = cell(4,1);
idy = 1;
IV = zeros(length(TMat),length(etaV));
for i=1:length(etaV)
    etaTemp = etaV(i);
    for idx = 1:1:length(TMat)
        T = TMat(idx);
        cf        = @(u) ChFBSHW(u, T, P0T, lambda, etaTemp, rho, sigma);
        valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
        frwdStock = S0 / P0T(T);
        valCOSFrwd = valCOS / P0T(T);
        IV(idx,idy) = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock,0.3);
    end
    argLegend{i} = sprintf('eta=%.3f',etaTemp);
    idy = idy +1;
end
MakeFigure(TMat, IV*100,argLegend,'Implied ATM volatility for parameter \eta')

% Effect of sigma

sigmaV   = [0.1,0.2,0.3,0.4];
argLegend = cell(4,1);
idy = 1;
IV = zeros(length(TMat),length(sigmaV));
for i=1:length(sigmaV)
    sigmaTemp = sigmaV(i);
    for idx = 1:1:length(TMat)
        T = TMat(idx);
        cf        = @(u) ChFBSHW(u, T, P0T, lambda, eta, rho, sigmaTemp);
        valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
        frwdStock = S0 / P0T(T);
        valCOSFrwd = valCOS / P0T(T);
        IV(idx,idy) = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock,0.3);
    end
    argLegend{i} = sprintf('sigma=%.1f',sigmaTemp);
    idy = idy +1;
end
MakeFigure(TMat, IV*100,argLegend,'Implied ATM volatility for parameter \sigma')

% Effect of rho

rhoV   = [-0.7, -0.3, 0.3, 0.7];
argLegend = cell(4,1);
idy = 1;
IV = zeros(length(TMat),length(rhoV));
for i=1:length(rhoV)
    rhoTemp = rhoV(i);
    for idx = 1:1:length(TMat)
        T = TMat(idx);
        cf        = @(u) ChFBSHW(u, T, P0T, lambda, eta, rhoTemp, sigma);
        valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
        frwdStock = S0 / P0T(T);
        valCOSFrwd = valCOS / P0T(T);
        IV(idx,idy) = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock,0.3);
    end
    argLegend{i} = sprintf('rho_{x,y}=%.1f',rhoTemp);
    idy = idy +1;
end
MakeFigure(TMat, IV*100,argLegend,'Implied ATM volatility for parameter \rho_{x,r}')

function cfV = ChFBSHW(u, T, P0T, lambd, eta, rho, sigma)

% Time step 

dt = 0.0001;

% Complex number

i = complex(0.0,1.0);
f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
   
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);  
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));  
C = @(u,tau)1.0/lambd*(i*u-1.0)*(1.0-exp(-lambd*tau));
 
 % Define a grid for the numerical integration of function theta

 zGrid = linspace(0.0,T,2500);
 term1 = @(u) 0.5*sigma*sigma *i*u*(i*u-1.0)*T;
 term2 = @(u) i*u*rho*sigma*eta/lambd*(i*u-1.0)*(T+1.0/lambd *(exp(-lambd*T)-1.0));
 term3 = @(u) eta*eta/(4.0*power(lambd,3.0))*power(i+u,2.0)*(3.0+exp(-2.0*lambd*T)-4.0*exp(-lambd*T)-2.0*lambd*T);
 term4 = @(u)  lambd*trapz(zGrid,theta(T-zGrid).*C(u,zGrid));
 A= @(u) term1(u) + term2(u) + term3(u) + term4(u);
 
 % Note that we don't include the B(u)*x0 term as it is included in the COS method

 cf = @(u)exp(A(u) + C(u,T)*r0 );
 
 % Iterate over all u and collect the ChF, iteration is necessary due to the integration in term4

 cfV = zeros(1,length(u));
 idx = 1;
 for ui=u
    cfV(idx)=cf(ui);
    idx = idx +1;
 end

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

k = 0:N-1;              % Row vector, index for expansion terms
u = k * pi / (b - a);   % ChF arguments

H_k = CallPutCoefficients('P',a,b,k);
temp = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);  % Multiply the first element by 1/2

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

function value=BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1 = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2 = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
 value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
 value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function impliedVol = ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock,initialVol)
func = @(sigma) (BS_Call_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice).^1.0;
impliedVol = fzero(func,initialVol);
 
function MakeFigure(X1, YMatrix1, argLegend,titleIn)

%CREATEFIGURE(X1,YMATRIX1)
%  X1:        Vector of x data
%  YMATRIX1:  Matrix of y data

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
%  'LineWidth',1,...
%  'Color',[0 0 0]);

plot1 = plot(X1,YMatrix1,'Parent',axes1,...
 'LineWidth',1.5);
set(plot1(1),'Marker','diamond','DisplayName',argLegend{1});
set(plot1(2),'Marker','square','LineStyle','-.',...
 'DisplayName',argLegend{2});
set(plot1(3),'Marker','o','LineStyle','-.','DisplayName',argLegend{3});
set(plot1(4),'DisplayName',argLegend{4});

% Create xlabel

xlabel({'T- time to maturity'});

% Create ylabel

ylabel({'implied volatility [%]'});

% Create title

title(titleIn);

% Create legend

legend1 = legend(axes1,'show');
set(legend1,'Color',[1 1 1]);
