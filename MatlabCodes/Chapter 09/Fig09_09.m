function CIR_paths
clc;close all;

% CIR model parameter settings

v0        = 0.1;
kappa     = 0.5;
vbar      = 0.1;
gamma     = 0.1;
NoOfPaths = 10; 
NoOfSteps = 500; 
T         = 5;

% Generation of the CIR paths with truncation scheme

[V,timeGrid] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,v0,kappa,vbar,gamma,T);

% Generate figure

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Density plot for S(t)

plot(timeGrid,V,'linewidth',1,'color',[0 0.45 0.74])
pdf = CIRDensity(kappa,gamma,vbar,0.0,T,v0);

% Grid for the densities of CIR

Tgrid=linspace(0.1,T,5);

x_arg=linspace(0,max(max(V(:,:)))*2,250);
for i=1:length(Tgrid)    
    plot3(Tgrid(i)*ones(length(x_arg),1),x_arg,pdf(x_arg),'k','linewidth',2)
end
axis([0,Tgrid(end),0,max(max(V))])
grid on;
xlabel('t')
ylabel('V(t)')
zlabel('CIR density')
view(axes1,[-68.8 40.08]);

function pdf = CIRDensity(kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
pdf      = @(x)1/c*ncx2pdf(x./c,d,kappaBar);

function [V,time] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,V0,kappa,vbar,gamma,T)

% Define initial value

V=zeros(NoOfPaths,NoOfSteps);
V(:,1) = V0;

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
