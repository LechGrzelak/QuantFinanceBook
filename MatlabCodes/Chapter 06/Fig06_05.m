function VG_Density_Recovery
close all;clc;
format long
i = complex(0,1); %assigning i=sqrt(-1)

% Characteristic function parameters 

S0    = 100;
r     = 0.1;
sigma = 0.12;
beta  = 0.2;
theta = -0.14;
t     = 1.0;

% Domain for the density f(x)

x = linspace(0.5,10,500);

% Characteristic function  

omega  = 1/beta*log(1-theta*beta -0.5*sigma^2*beta);
mu     = r+omega;
varPhi = @(u) (1-i*u*theta*beta + 0.5*sigma^2*beta*u.^2).^(-t/beta);
cf     = @(u) exp(i*u*mu*t + i*u*log(S0)).*varPhi(u);

% Cumulants needed for the integration range

zeta1 = (log(S0) -omega +theta)*t;
zeta2 = (sigma^2+beta*theta^2)*t;
zeta4 = 3*(sigma^4*beta+2*theta^4*beta^3+4*sigma^2*theta^2*beta^2)*t;

% Define the COS method integration range

L = 150;
a = zeta1 - L * sqrt(zeta2 + sqrt(zeta4));
b = zeta1 + L * sqrt(zeta2 + sqrt(zeta4));

% Reference value

f_XExact = COSDensity(cf,x,2^14,a,b);

% Iterate over different numbers of expansion terms

Ngrid = [ 16, 32, 64, 4096];
idx = 1;
error = zeros(length(Ngrid),1);
figure(1); hold on; grid on;xlabel('x');ylabel('f_X(x)')
legend_txt= cell(length(Ngrid),1);
for N = Ngrid

    % Density from the COS method

    f_X = COSDensity(cf,x,N,a,b);
     
    % Error

    error(idx) = max(f_XExact-f_X);
    plot(x,f_X,'linewidth',1.5)
    legend_txt{idx} = sprintf('N=%i',N);
    idx = idx + 1;
end
legend(legend_txt)
error
title('Variance Gamma density recovery with the COS method')

function f_X = COSDensity(cf,x,N,a,b)
i = complex(0,1); %assigning i=sqrt(-1)
k = 0:N-1; 
u = k * pi / (b - a);

% F_k coefficients

F_k    = 2 / (b - a) * real(cf(u) .* exp(-i * u * a));
F_k(1) = F_k(1) * 0.5; % Multiply the first term

% Final calculation

f_X = F_k * cos(u' * (x - a));
