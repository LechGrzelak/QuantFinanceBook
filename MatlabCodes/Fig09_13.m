function BrownianBridge
clc;close all;

NoOfPaths  =6;
NoOfSteps  = 1000; 
T          = 2;
vt0        = 0.1;
vtN        = 0.2;
sigma1      =0.03;
sigma2      =0.01;

figure(1)
randn('seed',1)
[B,timeGrid] = GeneratePathsBrownianBridge(NoOfPaths,NoOfSteps,T,vt0,vtN,sigma1);
plot(timeGrid,B,'color',[0,0.45,0.75],'linewidth',1.5)
axis([0,T,min(min(B)),max(max(B))])
grid on
xlabel('time')
ylabel('v(t)')
a = axis;
hold on
plot([0,T],[vt0,vtN],'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[1 0 0])

figure(2)
randn('seed',1)
[B,timeGrid] = GeneratePathsBrownianBridge(NoOfPaths,NoOfSteps,T,vt0,vtN,sigma2);
plot(timeGrid,B,'color',[0,0.45,0.75],'linewidth',1.5)
axis([0,T,min(min(B)),max(max(B))])
grid on
xlabel('time')
ylabel('v(t)')
axis(a)
hold on
plot([0,T],[vt0,vtN],'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[1 0 0])

function [B,time] = GeneratePathsBrownianBridge(NoOfPaths,NoOfSteps,T,a,b,sigma)

% Brownian bridge process approximated with the Euler discretization

B=zeros(NoOfPaths,NoOfSteps);
B(:,1) = a;

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
    B(:,i+1)  = B(:,i) +   (b-B(:,i))/(T -time(i))*dt + sigma*(W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end
