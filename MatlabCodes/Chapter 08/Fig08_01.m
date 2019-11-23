function CIR_Distribution
close all; clc

% Global parameters

s  =  0;
t  =  5;
x  =  0.001:0.001:0.5;

% Feller condition

q_Fcond =@(kappa,v_bar,gamma) 2*kappa*v_bar/gamma^2-1;

% Feller condition satsfied

gamma = 0.316;
kappa = 0.5;
v_s   = 0.2;
v_bar = 0.05;

[f_X,F_X]=  CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s);
q_F1 = q_Fcond(kappa,v_bar,gamma);

figure(1)
plot(x,f_X,'linewidth',1.5)
figure(2)
plot(x,F_X,'linewidth',1.5)

%% Feller condition not satisfied

gamma    = 0.129;
kappa    = 0.5;
v_s      = 0.2;
v_bar    = 0.05;
[f_X,F_X]=  CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s);
q_F2 = q_Fcond(kappa,v_bar,gamma);

figure(1)
hold on;plot(x,f_X,'--r','linewidth',1.5)
xlabel('x');ylabel('f(x)');legend(strcat('q_F=',num2str(round(q_F1,2))),strcat('q_F=',num2str(round(q_F2,2))))
title('Density for the CIR process')
axis([0,0.5,0,15])
grid on
figure(2)
hold on;plot(x,F_X,'--r','linewidth',1.5)
xlabel('x');ylabel('F(x)');legend(strcat('q_F=',num2str(round(q_F1,2))),strcat('q_F=',num2str(round(q_F2,2))))
title('Cumulative distribution function for the CIR process')
grid on

function [f_X,F_X]=  CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s)
c_s_t      = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d          = 4*kappa*v_bar/gamma^2;
lambda_s_t = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
f_X = 1/c_s_t*ncx2pdf(x/c_s_t,d,lambda_s_t);
F_X = ncx2cdf(x/c_s_t,d,lambda_s_t);
