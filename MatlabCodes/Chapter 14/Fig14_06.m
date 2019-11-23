function HW_negative_paths
clc;close all;
NoOfPaths = 45;
NoOfSteps = 200;
eta       = 0.04;
lambda    = 0.5;
T         = 1;
P0T       =@(T) exp(-0.02*T);
[r,time] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambda, eta);

%% First figure

title =['Hull-White, positive IR environment'];
PlotShortRate(r,time,lambda,eta,T,P0T,title)

%% Figure 2

eta       = 0.06;
lambda    = 0.5;
P0T       =@(T) exp(0.02*T);
[r,time] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambda, eta);
title =['Hull-White, negative IR environment'];
PlotShortRate(r,time,lambda,eta,T,P0T,title)

function [R,time] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambda, eta)

% Time step needed 

dt = 0.0001;
f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
   
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);  
theta = @(t) 1.0/lambda * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambda*lambda)*(1.0-exp(-2.0*lambda*t));  

% Define initial values

R=zeros(NoOfPaths,NoOfSteps);
R(:,1) = r0;

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
    R(:,i+1)  = R(:,i) + lambda*(theta(time(i))-R(:,i))*dt+  eta* (W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end

function r_mean=HWMean(P0T,lambd,eta,T)

% Time step needed

dt = 0.0001;
f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);

% Initial interest rate is forward rate at time t->0

r0     = f0T(0.00001);
theta  = Theta(P0T,eta,lambd);
zGrid  = linspace(0.0,T,2500);
temp   = @(z) theta(z) .* exp(-lambd*(T-z));
r_mean = r0*exp(-lambd*T) + lambd * trapz(zGrid,temp(zGrid));

function r_var = HWVar(lambd,eta,T)
r_var = eta*eta/(2.0*lambd) *( 1.0-exp(-2.0*lambd *T));

function pdf = HWDensity(x,P0T,lambd,eta,T)
r_mean = HWMean(P0T,lambd,eta,T);
r_var = HWVar(lambd,eta,T);
pdf = normpdf(x,r_mean,sqrt(r_var));
    
function theta =Theta(P0T,eta,lambd)

% Time step needed

dt = 0.0001;
f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2.0*dt);
theta = @(t)1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));      

function PlotShortRate(r,time,lambda,eta,T,P0T,titleIn)
figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(time,r,'linewidth',1,'color',[0 0.45 0.74])
y=@(x,T)HWDensity(x,P0T,lambda,eta,T);

Ti=linspace(0,T,5);
for i=2:length(Ti)
    ER = HWMean(P0T,lambda,eta,Ti(i));
    VarR = HWVar(lambda,eta,Ti(i));
    q1= norminv(0.999,ER,sqrt(VarR));
    q2= norminv(0.001,ER,sqrt(VarR));
    x_1=linspace(q2,0,100);
    x_2=linspace(0,q1,100);
    plot3(Ti(i)*ones(length(x_1),1),x_1,y(x_1,Ti(i)),'r','linewidth',2)
    plot3(Ti(i)*ones(length(x_2),1),x_2,y(x_2,Ti(i)),'k','linewidth',2)
    plot3([Ti(i),Ti(i)],[0,0],[0,y(x_1(end),Ti(i))],'--r','linewidth',2)
end
axis([0,T(end),q2,q1])
grid on;
xlabel('t')
ylabel('r(t)')
zlabel('density')
view(axes1,[-34.88 42.64]);
title(titleIn)
