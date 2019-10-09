function TimeChange
clc;close all;

% Heston model parameter settings

NoOfPaths = 25000; 
NoOfSteps = 500; 
T         = 5.0;

% Generate Brownian motion W(t)

[W,time] = GenerateBrownianMotion(NoOfPaths,NoOfSteps,T);

figure(1)
plot(time,W(1:20,:),'b');
hold on
grid on
legend('W(t)')

% Time-change function

nu = @(t) 0.3+ log(t+0.01)/10.0;

% Here we compute X(t) = nu(t) * W(t)

X=zeros(size(W));
for i =1:length(time)
    X(:,i) = nu(time(i)) * W(:,i);
end

figure(2)
plot(time,X(1:20,:),'color',[0,0.45,0.75],'linewidth',1.2);
hold on
grid on
legend('X(t)=nu(t)*W(t)')

% Here we compute Y(t) = W(nu^2(t)*t)

Y=zeros(size(W));
for i = 1:length(time)
    v = nu(time(i)).^2.*time(i);
    [~,idx] = min(abs(time-v));
    Y(:,i) = W(:,idx);
end
plot(time,Y(1:20,:),'r','linewidth',1.2);
hold on
grid on
legend('X(t)=W(nu(t)^2*t)')

% Comparison of the densities at time T

[y1,x1] = ksdensity(X(:,end));
[y2,x2] = ksdensity(Y(:,end));
figure(3)
plot(x1,y1,'b','linewidth',2);hold on
plot(x2,y2,'--r', 'linewidth',2)
grid on
legend('X(t)=nu(t)W(t)','Y(t)=W(mu^2(t)*t)')
function [W,time] = GenerateBrownianMotion(NoOfPaths,NoOfSteps,T)

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
    time(i+1) = time(i) + dt;
end
