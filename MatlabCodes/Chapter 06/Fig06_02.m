function COS_LogNormal_Density_Recovery
close all;clc;
format long
i = complex(0,1); %assigning i=sqrt(-1)

% COS method settings

a = -10; % Truncation domain
b =  10;

% Characteristic function parameters

mu     = 0.5;    sigma  = 0.2;

% Domain for the density f(x)

y = linspace(0.05,5,1000);

% Characteristic function  

cf  =  @(u) exp(i * mu * u - 0.5 * sigma^2 * u.^2);

% Iterate over different numbers of expansion terms

Ngrid = [16,64,128];
idx = 1;
figure(1); hold on; grid on;xlabel('y');ylabel('f_Y(y)')
legend_txt= cell(length(Ngrid),1);

for N=Ngrid

    % Density from the COS method

    f_Y =1./y .* COSDensity(cf,log(y),N,a,b);
    plot(y,f_Y)
    legend_txt{idx} = sprintf('N=%i',N);
    idx = idx + 1;
end
legend(legend_txt)

function f_X = COSDensity(cf,x,N,a,b)
i = complex(0,1); %assigning i=sqrt(-1)
k = 0:N-1; 
u = k * pi / (b - a);

% F_k coefficients

F_k    = 2 / (b - a) * real(cf(u) .* exp(-i * u * a));
F_k(1) = F_k(1) * 0.5; % adjustment for the first term

% Final calculation

f_X = F_k * cos(u' * (x - a));
