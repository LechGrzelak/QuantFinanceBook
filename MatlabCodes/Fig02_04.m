function LognormalDensityParameterEffects
close all; clc

% Effect of mu on the normal PDF with fixed sigma=1

figure(1)
hold on;
sigma = 1;

leg={5};
x=[0:0.01:5];
ind=0;
for mu=-1:0.5:1
    plot(x,lognpdf(x,mu,sigma),'linewidth',1.5)
    ind=ind+1;
    leg{ind}=strcat('$$\mu=',num2str(mu),'$$');
end

grid on
xlabel('x')
ylabel('PDF')
legendObj=legend(leg);
set(legendObj,'interpreter','latex')

% Effect of sigma on the normal PDF with fixed mu=0

figure(2)
hold on;
mu = 0;

leg={4};
ind=0;
for sigma=0.25:0.5:2
    plot(x,lognpdf(x,mu,sigma),'linewidth',1.5)
    ind=ind+1;
    leg{ind}=strcat('$$\sigma=',num2str(sigma),'$$');
end

grid on
xlabel('x')
ylabel('PDF')
legendObj=legend(leg);
set(legendObj,'interpreter','latex')
