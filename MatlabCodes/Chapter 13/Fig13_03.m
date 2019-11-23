function E_sqrt_V
clc;close all;

% Heston model parameter settings

NoOfPaths = 2500; 
NoOfSteps = 500; 
T         = 5.0;

% Parameters: kappa, gamma, v0, vbar

Parameters = [ 1.2, 0.1, 0.03, 0.04;
               1.2, 0.1, 0.035, 0.02;
               1.2, 0.2, 0.05, 0.02;
               0.8, 0.25, 0.15, 0.1;
               1.0, 0.2, 0.11, 0.06];
[n1,~]=size(Parameters);

figure(1);
hold on;
for i = 1 : n1
    kappa = Parameters(i,1);
    gamma = Parameters(i,2);
    v0 = Parameters(i,3);
    vbar = Parameters(i,4);   
[V,timeGrid] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,v0,kappa,vbar,gamma,T);
volHeston = sqrt(V);
    plot(timeGrid,mean(volHeston),'b','linewidth',2.0)
    approx1 = meanSqrtV_1(kappa,v0,vbar,gamma);
    approx2 = meanSqrtV_2(kappa,v0,vbar,gamma);
    timeGrid2 =linspace(0,T,20);
    plot(timeGrid2,approx1(timeGrid2),'--r')
    plot(timeGrid2,approx2(timeGrid2),'ok')
end
legend('Monte Carlo','Approximation 1', 'Approximation 2')
grid on
xlabel('time, t')
ylabel('Esqrt(V(t))')

function value = meanSqrtV_1(kappa,v0,vbar,gamma)
    delta = 4.0 *kappa*vbar/gamma/gamma;
    c= @(t) 1.0/(4.0*kappa)*gamma*gamma*(1.0-exp(-kappa*t));
    kappaBar = @(t) 4.0*kappa*v0*exp(-kappa*t)./(gamma*gamma*(1.0-exp(-kappa*t)));
    value= @(t) sqrt(c(t) .*(kappaBar(t)-1.0 +delta + delta./(2.0*(delta + kappaBar(t)))));
    
function value = meanSqrtV_2(kappa,v0,vbar,gamma)
    a = sqrt(vbar-gamma^2/(8.0*kappa));
    b = sqrt(v0)-a;
    temp = meanSqrtV_1(kappa,v0,vbar,gamma);
    epsilon1 = temp(1);
    c = -log(1.0/b  *(epsilon1-a));
    value= @(t) a + b *exp(-c*t);

function [V,time] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,v0,kappa,vbar,gamma,T)

% Define initial values

V=zeros(NoOfPaths,NoOfSteps);
V(:,1) = v0;

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
    V(:,i+1)  = V(:,i) + kappa*(vbar-V(:,i))*dt+  gamma* sqrt(V(:,i)).* (W(:,i+1)-W(:,i));
        
    % We apply here the truncation scheme to deal with negative values

    V(:,i+1) = max(V(:,i+1),0);
    time(i+1) = time(i) + dt;
end
