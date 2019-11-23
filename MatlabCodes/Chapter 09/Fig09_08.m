function ScatterPlot3DSurface
clc;close all;

% Monte Carlo model settings

NoOfPaths = 1000; 

mu1    = 5;
mu2    = 10;
sigma1 = 2;
sigma2 = 0.5;
rho    = 0.7;

% Generation of standard normal process

randn('seed',2)
Z1 = random('normal',0,1,[NoOfPaths,1]);
Z2 = random('normal',0,1,[NoOfPaths,1]);
Z1   = (Z1 - mean(Z1)) / std(Z1);
Z2   = (Z2 - mean(Z2)) / std(Z2);

% Correlation 

Z2 = rho*Z1 + sqrt(1-rho^2)*Z2;

% Adjustment for the mean and variance

Z1 = sigma1* Z1 + mu1;
Z2 = sigma2* Z2 + mu2;
        
% Generation of correlated BM, positive correlation

figure1 = figure;

% Create axes

axes1 = axes('Parent',figure1);
figure(1)
hold on;
grid on;
plot(Z1,Z2,'.','linewidth',1.5)
xlabel('X')
ylabel('Y')

% First marginal distribution

x1 =  norminv(0.001,mu1,sigma1)-1:0.1:norminv(1-0.001,mu1,sigma1)+1;
y1 = min(Z2)-1+ zeros([length(x1),1]);
z1 = normpdf(x1,mu1,sigma1);
fill3(x1,y1,z1, 'b', 'FaceAlpha', 1.0)

% Second marginal distribution

x2 =  norminv(0.001,mu2,sigma2)-1:0.1:norminv(1-0.001,mu2,sigma2)+1;
y2 = min(Z1)-1+ zeros([length(x2),1]);
z2 = normpdf(x2,mu2,sigma2);
fill3(y2,x2,z2, 'r', 'FaceAlpha', 1.0)
view(axes1,[119.6 37.2]);
legend('samples','f_X(x)','f_Y(y)')
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.165571430738749 0.758333337590808 0.171428569033742 0.15476190050443]);
