function MertonProcess_paths_3d
clc;close all;

% Monte Carlo settings

NoOfPaths = 200;
NoOfSteps = 500;
T         = 5;

% Model parameters

r     = 0.05;
S0    = 100;
sigma = 0.2;
sigmaJ= 0.5;
muJ   = 0;
xiP   = 1;

% Compute ChF for the Merton model

cf = ChFForMertonModel(r,T,muJ,sigmaJ,sigma,xiP,S0);

% Define the COS method integration range

L = 8;
a = - L * sqrt(T);
b = + L * sqrt(T);

% Domain for the density f(x)

x = linspace(-2,12,250);

% Exact representation 

f_XExact = COSDensity(cf,x,2^14,a,b);

% Monte Carlo paths for the Merton process

[S,time,X] = GeneratePathsMerton(NoOfPaths,NoOfSteps,S0,T,xiP,muJ,sigmaJ,r,sigma);

figure(1)
plot(time,X,'color',[0 0.45 0.74],'linewidth',1.2)
xlabel('time')
ylabel('$$X(t)$$','interpreter','latex')
grid on

figure(2)
plot(time,S,'color',[0 0.45 0.74],'linewidth',1.2)
xlabel('time')
ylabel('$$S(t)$$','interpreter','latex')
grid on

% Martingale error

error = mean(S(:,end))-S0*exp(r*T)

% Plot densities, from MC simulation and from COS method

[yCDF,xCDF]=ksdensity(X(:,end));
figure(3)
plot(xCDF,yCDF,'-r','linewidth',2.0)
hold on
plot(x,f_XExact,'--b','linewidth',2.0)
grid on
xlabel('x')
ylabel('PDF')
legend('Monte Carlo','COS method')

%%%%%%%%%%%%%%%%%% 3D figure for X(t) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3D density for X(t)

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(time,X(1:15,:),'linewidth',1,'color',[0 0.45 0.74])

% Grid for the densities of Merton model

Tgrid=linspace(1,T,5);

% Domain for the density f(x)

x_arg=linspace(0,max(max(X(:,:))),250);
mm=0;
for i=1:length(Tgrid)    

    % Compute ChF for the Merton model

    cf = ChFForMertonModel(r,Tgrid(i),muJ,sigmaJ,sigma,xiP,S0);
    L = 8;
    a = - L * sqrt(T);
    b = + L * sqrt(T);
    f_COS = COSDensity(cf,x_arg,2^14,a,b);
    plot3(Tgrid(i)*ones(length(x_arg),1),x_arg,f_COS,'k','linewidth',2)
    mm=max(mm,max(f_COS));
end
axis([0,Tgrid(end),0,max(max(X))])
grid on;
xlabel('t')
ylabel('X(t)')
zlabel('Merton density')
view(axes1,[-68.8 40.08]);
axis([0,T,min(min(x_arg))-1,max(max(x_arg))+1,0,mm])

%%%%%%%%%%%%%%%%%% 3D figure for S(t) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3D density

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(time,S(1:15,:),'linewidth',1,'color',[0 0.45 0.74])

% Grid for the densities of Merton model

Tgrid=linspace(1,T,5);

% Domain for the density f(x)

x_arg=linspace(0,max(max(X(:,:))),250);
mm=0;
for i=1:length(Tgrid)    

    % Compute ChF for the Merton model

    cf = ChFForMertonModel(r,Tgrid(i),muJ,sigmaJ,sigma,xiP,S0);
    L = 8;
    a = - L * sqrt(Tgrid(i));
    b = + L * sqrt(Tgrid(i));
    f_COS = 1./exp(x_arg).* COSDensity(cf,x_arg,2^14,a,b);
    plot3(Tgrid(i)*ones(length(exp(x_arg)),1),exp(x_arg),f_COS,'k','linewidth',2)
    mm=max(mm,max(f_COS));
end
axis([0,Tgrid(end),0,max(max(S0*exp(r*T)*10))])
grid on;
xlabel('t')
ylabel('S(t)')
zlabel('Merton density')
view(axes1,[-68.8 40.08]);

function cf = ChFForMertonModel(r,tau,muJ,sigmaJ,sigma,xiP,S0)

% Term for E(exp(J)-1)

i = complex(0,1);
helpExp = exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0;
  
% Characteristic function for Merton's model    

cf = @(u) exp(i*u*log(S0)).*exp(i * u .* (r - xiP * helpExp - 0.5 * sigma * sigma) *tau...
        - 0.5 * sigma * sigma * u.^2 * tau + xiP * tau .* ...
        (exp(i * u * muJ - 0.5 * sigmaJ * sigmaJ * u.^2)-1.0));

function f_X = COSDensity(cf,x,N,a,b)
i = complex(0,1); %assigning i=sqrt(-1)
k = 0:N-1; 
u = k * pi / (b - a);

% F_k coefficients

F_k    = 2 / (b - a) * real(cf(u) .* exp(-i * u * a));
F_k(1) = F_k(1) * 0.5; % Multiply the first term

% Final calculation

f_X = F_k * cos(u' * (x - a));

    
function [S,time,X] =GeneratePathsMerton(NoOfPaths,NoOfSteps,S0,T,xiP,muJ,sigmaJ,r,sigma)

% Empty matrices for the Poisson process and stock paths

Xp     = zeros(NoOfPaths,NoOfSteps);
X      = zeros(NoOfPaths,NoOfSteps);
W      = zeros(NoOfPaths,NoOfSteps);
X(:,1) = log(S0);
S      = zeros(NoOfPaths,NoOfSteps);
S(:,1) = S0;
dt        = T/NoOfSteps;

% Random noise

Z1 = random('poisson',xiP*dt,[NoOfPaths,NoOfSteps]);
Z2 = random('normal',0,1,[NoOfPaths,NoOfSteps]);
J  = random('normal',muJ,sigmaJ,[NoOfPaths,NoOfSteps]);

% Creation of the paths

% Expectation E(exp(J))

EeJ = exp(muJ + 0.5*sigmaJ^2);
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps  
    if NoOfPaths>1
        Z2(:,i) = (Z2(:,i)-mean(Z2(:,i)))/std(Z2(:,i));
    end
    Xp(:,i+1) = Xp(:,i) + Z1(:,i);
    W(:,i+1)  = W(:,i)  + sqrt(dt)* Z2(:,i);
        
    X(:,i+1)  = X(:,i) + (r- xiP*(EeJ-1)-0.5*sigma^2)*dt +  sigma* (W(:,i+1)-W(:,i)) + J(:,i).*(Xp(:,i+1)-Xp(:,i));
    S(:,i+1) = exp(X(:,i));
    time(i+1) = time(i) + dt;
end

