function StockPathAndGreeks
clc; close all;
randn('seed',43) 

% Monte Carlo settings

s0        = 10;
noOfPaths = 50;
noOfSteps = 500;
r         = 0.05;
sigma     = 0.4;
T         = 1.001;

% Strike price

K         = 10;

% Path to plot

pathId    = 6;

[S,x] = GeneratePathsGBMEuler(noOfPaths,noOfSteps,T,r,sigma,s0);

% Prepare a few function pointers

CallOpt   = @(t,s0)BS_Call_Put_Option_Price('c',s0,K,sigma,t,T,r);
PutOpt    = @(t,s0)BS_Call_Put_Option_Price('p',s0,K,sigma,t,T,r);
DeltaCall = @(t,s0)BS_Delta('c',s0,K,sigma,t,T,r);
Gamma     = @(t,s0)BS_Gamma(s0,K,sigma,t,T,r);
Vega      = @(t,s0)BS_Vega(s0,K,sigma,t,T,r);

% Grid for the stock values 

s0Grid     = linspace(s0/100,1.5*s0,50);
timeGrid   = 0.02:0.025:T;

% Prepare empty matrices for storing the results

callOptM   = zeros(length(timeGrid),length(s0Grid));
putOptM    = zeros(length(timeGrid),length(s0Grid));
deltaCallM = zeros(length(timeGrid),length(s0Grid));
gammaM     = zeros(length(timeGrid),length(s0Grid));
vegaM      = zeros(length(timeGrid),length(s0Grid));
TM         = zeros(length(timeGrid),length(s0Grid));
s0M        = zeros(length(timeGrid),length(s0Grid));

for i=1:length(timeGrid)
    TM(i,:)         = timeGrid(i)*ones(length(s0Grid),1);
    s0M(i,:)        = s0Grid;
    callOptM(i,:)   = CallOpt(timeGrid(i),s0Grid);
    putOptM(i,:)    = PutOpt(timeGrid(i),s0Grid);
    deltaCallM(i,:) = DeltaCall(timeGrid(i),s0Grid); 
    gammaM(i,:)     = Gamma(timeGrid(i),s0Grid); 
    vegaM(i,:)      = Vega(timeGrid(i),s0Grid); 
end

% Stock path

figure(1)
plot(x,S(pathId,:),'LineWidth',1.5)
xlabel('t')
ylabel('S(t)')
axis([0,T,min(S(pathId,:)),max(S(pathId,:))])
grid on

% Call option surface with Monte Carlo path

figure(2)
mesh(TM,s0M,callOptM, 'edgecolor', [0.75,0.45,0])
grid on
hold on

% F stores the time grid - stock grid - with call prices

F = griddedInterpolant(TM,s0M,callOptM);
V = zeros(length(x),1);
for j=1:length(x)
    V(j) = F(x(j),S(pathId,j));
end
plot3(x,S(pathId,:),V,'r','linewidth',2)
axis([0    T       0  max(s0Grid)       0  max(max(callOptM))])
xlabel('t')
ylabel('S')
zlabel('V')
title('Call option')

% Put option surface with Monte Carlo path

figure(3)
mesh(TM,s0M,putOptM, 'edgecolor', [0.75,0.45,0])
grid on
hold on
F = griddedInterpolant(TM,s0M,putOptM);
for j=1:length(x)
    V(j) = F(x(j),S(pathId,j));
end
plot3(x,S(pathId,:),V,'r','linewidth',2)
axis([0    T       0  max(s0Grid)       0  max(max(putOptM))])
xlabel('t')
ylabel('S')
zlabel('V')
title('Put Option')

% Call option delta profile

figure(4) 
mesh(TM,s0M,deltaCallM, 'edgecolor', [0.75,0.45,0])
grid on
hold on
F = griddedInterpolant(TM,s0M,deltaCallM);
for j=1:length(x)
    V(j) = F(x(j),S(pathId,j));
end
plot3(x,S(pathId,:),V,'r','linewidth',2)
axis([0    T       0  max(s0Grid)       0  max(max(deltaCallM))])
xlabel('t')
ylabel('S')
zlabel('Delta')
title('Delta for a call option')

% Option gamma profile

figure(5)
mesh(TM,s0M,gammaM, 'edgecolor', [0.75,0.45,0])
grid on
hold on
F=griddedInterpolant(TM,s0M,gammaM);
for j=1:length(x)
    V(j) = F(x(j),S(pathId,j));
end
plot3(x,S(pathId,:),V,'r','linewidth',2)
axis([0    T       0  max(s0Grid)       0  max(max(gammaM))])
xlabel('t')
ylabel('S')
zlabel('Gamma')
title('Gamma')

% Option vega profile

figure(6)
mesh(TM,s0M,vegaM, 'edgecolor', [0.75,0.45,0])
grid on
hold on
F=griddedInterpolant(TM,s0M,vegaM);
for j=1:length(x)
    V(j) = F(x(j),S(pathId,j));
end
plot3(x,S(pathId,:),V,'r','linewidth',2)
axis([0    T       0  max(s0Grid)       0  max(max(vegaM))])
xlabel('t')
ylabel('S')
zlabel('Vega')
title('Vega')

% Black-Scholes call option price

function value = BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
d2    = d1 - sigma * sqrt(T-t);
if lower(CP) == 'c' || lower(CP) == 1
    value = normcdf(d1) .* S_0 - normcdf(d2) .* K * exp(-r * (T-t));
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(-d2) .* K*exp(-r*(T-t)) - normcdf(-d1).*S_0;
end

function value = BS_Delta(CP,S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
if lower(CP) == 'c' || lower(CP) == 1
   value = normcdf(d1);
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(d1)-1;
end
    
function value = BS_Gamma(S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
value = normpdf(d1)./(S_0*sigma*sqrt(T-t));

function value = BS_Vega(S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
value = S_0.*normpdf(d1)*sqrt(T-t);


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
