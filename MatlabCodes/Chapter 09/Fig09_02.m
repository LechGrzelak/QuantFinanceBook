function MonteCarloIntegral2
close all;

% Monte Carlo settings

NoOfPaths=2000;         
NoOfSteps=100;          
T    = 2; dt=T/NoOfSteps;
X    = zeros(NoOfPaths,NoOfSteps);
W    = zeros(NoOfPaths,NoOfSteps);

% Generate set of random numbers

Z    = random('normal',0,1,[NoOfPaths,NoOfSteps]);

% Run simulation

t_i = 0;
for i=1:NoOfSteps
    if NoOfPaths>1
        Z(:,i) = (Z(:,i)-mean(Z(:,i)))/std(Z(:,i));
    end
    W(:,i+1) = W(:,i) + sqrt(dt) * Z(:,i);
    X(:,i+1) = X(:,i) + W(:,i) .* (W(:,i+1)-W(:,i));
    t_i = i*T/NoOfSteps;
end

% Expectation and variance

EX = mean(X(:,end));
VarX = var(X(:,end));
sprintf('E(X(T)) =%f and Var(X(T))= %f',EX,VarX)

% Figure 1

figure1 = figure(1);
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(0:dt:T,X,'Color',[0,0.45,0.75])
grid on;
xlabel('time')
ylabel('X(t)')

% Figure 2

figure2 = figure(2);
axes2 = axes('Parent',figure2);
hold(axes2,'on');

h1 = histogram(X(:,end),50);
h1.FaceColor=[1,1,1];
xlabel('X(T)')
title('Histogram of X(T) at T = 2')
grid on;
    
