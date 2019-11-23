function ChF_3D
close all;clc
mu    = 10;
sigma = 1;
i     = complex(0,1);
chf  = @(u)exp(i * mu * u - sigma * sigma * u .* u / 2.0);
pdf  = @(x)normpdf(x,mu,sigma);
cdf  = @(x)normcdf(x,mu,sigma);

% Density 

figure(1)
x = linspace(5,15,100);
plot(x,pdf(x),'linewidth',2)
grid on
xlabel('x')
ylabel('PDF')

% Cumulative distribution function

figure(2)
plot(x,cdf(x),'linewidth',2)
grid on
xlabel('x')
ylabel('CDF')

% 3D graph for the ChF

figure(3)
u = linspace(0,5,250);
chfVal = chf(u);
plot3(u,real(chfVal),imag(chfVal),'linewidth',2)
grid on
xlabel('u')
ylabel('real(ChF)')
zlabel('imag(ChF)')
