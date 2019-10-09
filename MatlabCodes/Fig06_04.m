function COS_CGMY_Density_Recovery
close all;clc;
format long
i = complex(0,1); %assigning i=sqrt(-1)

% Characteristic function parameters 

S0    = 100;
r     = 0.1;
sigma = 0.2;
C     = 1.0;
G     = 5.0;
M     = 5.0;
Y     = 1.5;
t     = 1.0;

% Domain for the density f(x)

x = linspace(-2,12,250);

% Characteristic function  

varphi = @(u)exp(t * C * gamma(-Y) * ((M-i*u).^Y - M^Y + (G+i*u).^Y - G^Y)); 
omega  = -1/t * log(varphi(-i));
cf     = @(u) exp(i*u*(log(S0)+r+omega-0.5*sigma^2)*t - 0.5*sigma^2*u.^2*t).*varphi(u);

% Cumulants needed for the integration range

zeta1 = (log(S0) + r + omega -0.5*sigma^2)*t + t*C*Y*gamma(-Y)*(G^(Y-1)-M^(Y-1));
zeta2 = sigma^2*t + t*C*gamma(2-Y)*(G^(Y-2)+M^(Y-2));
zeta4 = t*C*gamma(4-Y) * (G^(Y-4)+M^(Y-4));

% Define the COS method integration range

L = 8;
a = zeta1 - L * sqrt(zeta2 + sqrt(zeta4));
b = zeta1 + L * sqrt(zeta2 + sqrt(zeta4));

% Exact values 

f_XExact = COSDensity(cf,x,2^14,a,b);

% Iterate over different numbers of expansion terms

Ngrid = 2.^(2:1:5);
idx = 1;
error = zeros(length(Ngrid),1);
figure(1); hold on; grid on;xlabel('x');ylabel('f_X(x)')
legend_txt= cell(length(Ngrid),1);
time_txt= cell(length(Ngrid),1);
for N = Ngrid

    % Density from the COS method

    tic
    for j = 1:1000
        f_X = COSDensity(cf,x,N,a,b);
    end
    timeEnd = toc/j * 1000;

    % Error

    error(idx) = max(f_XExact-f_X);
    plot(x,f_X,'linewidth',1.5)
    legend_txt{idx} = sprintf('N=%i',N);
    time_txt{idx}= sprintf('time neeeded for evaluation = %.2f miliseconds',timeEnd);
    idx = idx + 1;
end
legend(legend_txt)
error
time_txt
title('CGMY density recovery with the COS method')

function f_X = COSDensity(cf,x,N,a,b)
i = complex(0,1); %assigning i=sqrt(-1)
k = 0:N-1; 
u = k * pi / (b - a);

% F_k coefficients

F_k    = 2 / (b - a) * real(cf(u) .* exp(-i * u * a));
F_k(1) = F_k(1) * 0.5; % adjustment for the first term

% Final calculation

f_X = F_k * cos(u' * (x - a));
