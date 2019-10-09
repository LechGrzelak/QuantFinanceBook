function COS_Normal_Density_Recovery
close all;clc;
format long
i = complex(0,1); %assigning i=sqrt(-1)

% The COS method settings

a = -10; % truncation domain
b =  10;

% Characteristic function parameters

mu     = 0;    sigma  = 1;

% Domain for the density f(x)

x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];

% Characteristic function  

cf  =  @(u) exp(i * mu * u - 0.5 * sigma^2 * u.^2);

% Iterate over different numbers of expansion terms

Ngrid = 2.^(2:1:6);
idx = 1;
error = zeros(length(Ngrid),1);
figure(1); hold on; grid on;xlabel('x');ylabel('f_X(x)')
for N=Ngrid

    % Density from the COS method

    f_X = COSDensity(cf,x,N,a,b);

    % Error

    error(idx) = max(normpdf(x)-f_X);
    idx = idx + 1;
    plot(x,f_X)
end

error

function f_X = COSDensity(cf,x,N,a,b)
i = complex(0,1); %assigning i=sqrt(-1)
k = 0:N-1; 
u = k * pi / (b - a);

% F_k coefficients

F_k    = 2 / (b - a) * real(cf(u) .* exp(-i * u * a));
F_k(1) = F_k(1) * 0.5; % adjustment for the first term

% Final calculation

f_X = F_k * cos(u' * (x - a));
