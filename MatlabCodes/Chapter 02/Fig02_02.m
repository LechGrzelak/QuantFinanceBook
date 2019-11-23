function GBM_ABM_paths
clc;close all;

% Initial model settings

S0        = 100;
NoOfPaths = 40;
NoOfSteps = 200;
r         = 0.05;
sigma     = 0.4;
T         = 1;
dt        = T/NoOfSteps;

S=zeros(NoOfPaths,NoOfSteps);
X=zeros(NoOfPaths,NoOfSteps);
S(:,1) = S0;
X(:,1) = log(S0);
randn('seed',4) 
Z=randn([NoOfPaths,NoOfSteps]);

for i=1:NoOfSteps
    Z(:,i)   = (Z(:,i) - mean(Z(:,i))) / std(Z(:,i));
    X(:,i+1) = X(:,i) + (r- 0.5*sigma^2)*dt+  sigma* sqrt(dt).*Z(:,i);
    S(:,i+1) = exp(X(:,i+1));
end
t= 0:dt:T;

%% First figure, for S(t)

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');

%density plot for S(t)

plot(t,S,'linewidth',1,'color',[0 0.45 0.74])
y=@(x,T)lognpdf(x,log(S0)+(r-0.5*sigma^2)*T,sigma*sqrt(T));
T=linspace(0,T,5);

for i=1:length(T)
    ind=find(t>=T(i),1,'first');
    x_arg=linspace(min(S(:,ind))-20,max(S(:,ind))+40,100);
plot3(T(i)*ones(length(x_arg),1),x_arg,y(x_arg,T(i)),'k','linewidth',2)
end
axis([0,T(end),0,max(max(S))])
grid on;
xlabel('t')
ylabel('S(t)')
zlabel('density')
view(axes1,[-68.8 40.08]);

%% First figure, for X(t)

figure2 = figure;
axes2 = axes('Parent',figure2);
hold(axes2,'on');

plot(t,X,'linewidth',1,'color',[0 0.45 0.74])
hold on;
y=@(x,T)normpdf(x,log(S0)+(r-0.5*sigma^2)*T,sigma*sqrt(T));
for i=1:length(T)
    ind=find(t>=T(i),1,'first');
    x_arg=linspace(min(X(:,ind))-1,max(X(:,ind))+1);
plot3(T(i)*ones(length(x_arg),1),x_arg,y(x_arg,T(i)),'k','linewidth',2)
end
axis([0,T(end),min(min(X))-0.5,max(max(X))+0.5])
grid on;
xlabel('t')
ylabel('X(t)')
zlabel('density')
view(axes2,[-68.8 40.08]);



