function MonteCarloIntegral
close all;
% Monte Carlo settings
NoOfPaths=2000;         
NoOfSteps=100;          
T    = 1; dt=T/NoOfSteps;
X    = zeros(NoOfPaths,NoOfSteps);

% Define the function to be integrated int_0^t g(t) dW(t)

g    = @(t) t.^2;

% Generate set of random numbers

Z    = random('normal',0,1,[NoOfPaths,NoOfSteps]);

% Run simulation

t_i = 0;
for i=1:NoOfSteps
    if NoOfPaths>1
        Z(:,i) = (Z(:,i)-mean(Z(:,i)))/std(Z(:,i));
    end
    X(:,i+1) = X(:,i) + g(t_i) * sqrt(dt) * Z(:,i);
    t_i = i*T/NoOfSteps;
end

% Expectation and variance

EX = mean(X(:,end));
VarX = var(X(:,end));
sprintf('E(X(T)) =%f and Var(X(T))= %f',EX,VarX)

% Figure 1

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(0:dt:T,X,'Color',[0,0.45,0.75])

% 3D graph with marginal density at T

mu    = 0;
sigma = sqrt(0.2);

% Define range for the density

x=-4*sigma:0.01:4*sigma;
plot3(T*ones(length(x),1),x,normpdf(x,mu,sigma),'linewidth',2,'color',[0,0,0])
grid on;
view(axes1,[-38.72 33.68]);
xlabel('time')
ylabel('X(t)')
zlabel('PDF')

% Figure 2

figure2 = figure;
axes2 = axes('Parent',figure2);
hold(axes2,'on');

hist(X(:,end),50,'FaceColor',[0 0 0],'EdgeColor',[0.15 0.15 0.15])
xlabel('X(T)')
title('Histogram of X(T) at T = 1')
grid on;
    
