function CorrelatedBM
clc;close all;

% Monte Carlo model settings

NoOfPaths = 1; 
NoOfSteps = 500; 
T         = 1;

% Generation of correlated BM, positive correlation

rho = 0.7;
randn('seed',2)
[W1,W2,time] = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho);
figure(1)
hold on;
grid on;
plot(time,W1,'linewidth',1.5)
plot(time,W2,'linewidth',1.5)
xlabel('time')
ylabel('W(t)')

% Generation of correlated BM, negative correlation

rho = -0.7;
randn('seed',2)
[W1,W2,time] = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho);
figure(2)
hold on;
grid on;
plot(time,W1,'linewidth',1.5)
plot(time,W2,'linewidth',1.5)
xlabel('time')
ylabel('W(t)')

% Generation of correlated BM, zero correlation

rho = 0.0;
randn('seed',2)
[W1,W2,time] = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho);
figure(3)
hold on;
grid on;
plot(time,W1,'linewidth',1.5)
plot(time,W2,'linewidth',1.5)
xlabel('time')
ylabel('W(t)')

function [W1,W2,time] = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho)

% Random noise

Z1=random('normal',0,1,[NoOfPaths,NoOfSteps]);
Z2=random('normal',0,1,[NoOfPaths,NoOfSteps]);

W1=zeros([NoOfPaths,NoOfSteps]);
W2=zeros([NoOfPaths,NoOfSteps]);

dt = T / NoOfSteps;
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps
    if NoOfPaths>1
        Z1(:,i)   = (Z1(:,i) - mean(Z1(:,i))) / std(Z1(:,i));
        Z2(:,i)   = (Z2(:,i) - mean(Z2(:,i))) / std(Z2(:,i));
    end
    Z2(:,i) = rho * Z1(:,i) + sqrt(1-rho^2)*Z2(:,i);

    % Correlated Brownian motions

    W1(:,i+1) = W1(:,i) + sqrt(dt).*Z1(:,i);
    W2(:,i+1) = W2(:,i) + sqrt(dt).*Z2(:,i);
    
    time(i+1) = time(i) + dt;
end
