function PathsUnderQandPmeasure
close all;
randn('seed',3)
NoOfPaths = 8;
NoOfSteps = 1000;

T  = 10;
dt = T/NoOfSteps;

% Normals

Z1 = random('normal',0,1,[NoOfPaths,NoOfSteps])*sqrt(dt);
Z2 = random('normal',0,1,[NoOfPaths,NoOfSteps])*sqrt(dt);

% Parameters

S_0   = 1;
r     = 0.05;
mu    = 0.15;
sigma = 0.1;

% Generation of the paths

S1      = zeros(NoOfPaths,NoOfSteps);
S1(:,1) = S_0;
S2      = zeros(NoOfPaths,NoOfSteps);
S2(:,1) = S_0;

for i=1:NoOfSteps
    Z1(:,i) = (Z1(:,i)- mean(Z1(:,i)))/std(Z1(:,i))*sqrt(dt);
    Z2(:,i) = (Z2(:,i)- mean(Z2(:,i)))/std(Z2(:,i))*sqrt(dt);
    
    S1(:,i+1)=S1(:,i)+mu*S1(:,i)*dt + sigma*S1(:,i).*Z1(:,i);
    S2(:,i+1)=S2(:,i)+r*S2(:,i)*dt + sigma*S2(:,i).*Z2(:,i);
end

%% Plotting the results

time=0:dt:T;
figure(1)
hold on
plot(time,S_0*exp(mu*time)./exp(r*time),'--r','linewidth',2)
plot(time,S1.*exp(-r*time),'linewidth',1,'color',[0 0.45 0.74])
a=axis;
a(3)=0;
axis(a);
grid on;
xlabel('time')
lab=ylabel('$$\frac{S(t)}{M(t)}$$');
set(lab,'Interpreter','latex');
leg = legend('$$E^P\Big[\frac{S(t)}{M(t)}\Big]$$','paths $$\frac{S(t)}{M(t)}$$');
set(leg,'Interpreter','latex');

plot(time,S_0*exp(mu*time)./exp(r*time),'--r','linewidth',2)
titleT=title('Monte Carlo Paths of $$\frac{S(t)}{M(t)}$$ under real-world measure, P');
set(titleT,'Interpreter','latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2)
hold on
plot(time,S_0*ones(length(time),1),'--r','linewidth',2)
plot(time,S2.*exp(-r*time),'linewidth',1,'color',[0 0.45 0.74])

a(3)=0;
axis(a);
grid on;
xlabel('time')
lab=ylabel('$$\frac{S(t)}{M(t)}$$');
set(lab,'Interpreter','latex');
leg = legend('$$E^Q\Big[\frac{S(t)}{M(t)}\Big]$$','paths $$\frac{S(t)}{M(t)}$$');
set(leg,'Interpreter','latex');

plot(time,S_0*ones(length(time),1),'--r','linewidth',2)
titleT=title('Monte Carlo Paths of $$\frac{S(t)}{M(t)}$$ under risk-neutral measure, Q');
set(titleT,'Interpreter','latex');
