function CIR_QE
close all;clc;

% The switching criterion 

aStar = 1.5;

% Number of Monte Carlo samples

NoOfSamples = 1000000;

% Maturity time

T = 3;

% CIR parameter settings

gamma  = 0.6; % 0.2
vbar   = 0.05;
v0     = 0.3;
kappa  = 0.5;

% Mean and the variance for the CIR process

m  = CIRMean(kappa,gamma,vbar,0,T,v0);
s2 = CIRVar(kappa,gamma,vbar,0,T,v0);

% QE simulation of the samples for v(T)|v(t_0)

if (s2/m^2 <aStar)
    %%% a and b- first approach
    [a,b] = FirstApproach(m,s2);
    Z = random('normal',0,1,[NoOfSamples,1]);
    Z = (Z-mean(Z))/std(Z);
    V = a * (b+Z).^2;
else
    %%%% c & d- second approach
    [c,d] = SecondApproach(m,s2);
    U=random('uniform',0,1,[NoOfSamples,1]);
    V = 1/d * log((1-c)./(1-U));
    [a1]=find(U<=c);
    V(a1) = 0;
end

% Mean and variance errors

mean_proxy = mean(V);
mean_error = mean_proxy - m;
var_proxy = var(V);
var_error = var_proxy - s2;
fprintf('mean error = %0.5f, var error =%.5f',mean_error,var_error);

% Plotting of the CDFs

figure(1); hold on; grid on

% Empirical and theoretical CDF 

cdf_theo       = CIRCDF(kappa,gamma,vbar,0,T,v0);
[cdfEmpir, x] = ecdf(V);
plot(x,cdf_theo(x),'-','linewidth',2,'color',[0,0.45,0.75])
plot(x,cdfEmpir,'--r','linewidth',2);
ylabel('CDF')
if (s2/m^2 <aStar)
    legend('F_{v(T)}(x)','F_{v_1(T)}(x)')
    axis([0,0.6,0,1])
else
    legend('F_{v(T)}(x)','F_{v_2(T)}(x)')
    axis([0,2,0,1])
end

function cdf = CIRCDF(kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
cdf      = @(x)ncx2cdf(x./c,d,kappaBar);

function pdf = CIRDensity(kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
pdf      = @(x)1/c*ncx2pdf(x./c,d,kappaBar);

function EV = CIRMean(kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
EV = c*(d+kappaBar);

function VarV = CIRVar(kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
VarV = c^2*(2*d+4*kappaBar);

function [a,b] = FirstApproach(m,s2)
b2 = 2*m^2/s2 - 1 + sqrt(2*m^2/s2)*sqrt(2*m^2/s2-1); 
b  = sqrt(b2);
a  = m /(1+b2);

function [c,d] = SecondApproach(m,s2)
c = (s2/m^2-1)/(s2/m^2+1);
d = (1-c)/m;
