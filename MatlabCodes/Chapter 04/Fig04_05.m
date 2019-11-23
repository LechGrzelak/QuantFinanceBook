function Call_dCdK_d2CdK2
clc; close all;

% Model settings

s0        = 10;
r         = 0.05;
sigma     = 0.4;
T         = 1.001;

% Prepare a few function pointers

CallOpt   = @(t,K)BS_Call_Option_Price('c',s0,K,sigma,t,T,r);
dCdKOpt   = @(t,K)dCdK(s0,K,sigma,t,T,r);
d2CdK2Opt = @(t,K)d2CdK2(s0,K,sigma,t,T,r);

% Grid for the stock values 

KGrid     = linspace(s0/100,1.5*s0,50);
timeGrid   = 0.02:0.025:T;

% Prepare empty matrices for storing the results

callOptM   = zeros(length(timeGrid),length(KGrid));
dCdKM      = zeros(length(timeGrid),length(KGrid));
d2CdK2M    = zeros(length(timeGrid),length(KGrid));
TM         = zeros(length(timeGrid),length(KGrid));
KM         = zeros(length(timeGrid),length(KGrid));

for i=1:length(timeGrid)
    TM(i,:)         = timeGrid(i)*ones(length(KGrid),1);
    KM(i,:)         = KGrid;
    callOptM(i,:)   = CallOpt(timeGrid(i),KGrid);
    dCdKM(i,:)      = dCdKOpt(timeGrid(i),KGrid);
    d2CdK2M(i,:)    = d2CdK2Opt(timeGrid(i),KGrid);
end

% Call option surface

figure1 = figure(1);
axes1 = axes('Parent',figure1);
hold(axes1,'on');

mesh(TM,KM,callOptM, 'edgecolor', [0.75,0.45,0])
grid on
hold on
axis([0    T       0  max(KGrid)       0  max(max(callOptM))])
xlabel('t')
ylabel('K')
zlabel('Call')
view(axes1,[-117.5 18.8]);

% dC / dK surface 

figure1 = figure(2);
axes1 = axes('Parent',figure1);
hold(axes1,'on');

mesh(TM,KM,dCdKM, 'edgecolor', [0.75,0.45,0])
grid on
hold on
axis([0    T       0  max(KGrid)       min(min(dCdKM))  0  ])
xlabel('t')
ylabel('K')
zlabel('dC / dK')
view(axes1,[-21.1 37.2]);

% d2CdK2 surface

figure1 = figure(3);
axes1 = axes('Parent',figure1);
hold(axes1,'on');

mesh(TM,KM,d2CdK2M, 'edgecolor', [0.75,0.45,0])
grid on
hold on
axis([0    T       0  max(KGrid)       0  max(max(d2CdK2M))])
xlabel('t')
ylabel('K')
zlabel('d2C / dK2')
view(axes1,[-117.5 18.8]);


% Black-Scholes call option price

function value = BS_Call_Option_Price(CP,S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
d2    = d1 - sigma * sqrt(T-t);
if lower(CP) == 'c' || lower(CP) == 1
    value = normcdf(d1) .* S_0 - normcdf(d2) .* K * exp(-r * (T-t));
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(-d2) .* K*exp(-r*(T-t)) - normcdf(-d1).*S_0;
end

function value = dCdK(S_0,K,sigma,t,T,r)
c = @(k)BS_Call_Option_Price('c',S_0,k,sigma,t,T,r);
dK = 0.0001;
value = (c(K+dK)-c(K-dK))/(2.0*dK);

function value = d2CdK2(S_0,K,sigma,t,T,r)
dCdK_ = @(k)dCdK(S_0,k,sigma,t,T,r);
dK = 0.0001;
value = (dCdK_(K+dK)-dCdK_(K-dK))/(2.0*dK);
