function BatesImpliedVolatilitySurface
clc;clf;close all;

% Option parameters

S0     = 100;
CP     = 'c';

% Heston model part

kappa  = 1.1;
gamma  = 0.8;
vBar   = 0.05;
rho    = -0.75;
v0     = 0.05;
r      = 0.00;

% Bates model

muJ    = 0.0;
sigmaJ = 0.0;
xiP    = 0.2;

% Range of strike prices

K      = linspace(100,200,15)'; 

figure(1)
hold on;
grid on;

% COS method settings

N = 5000;
L = 5;

TMat = linspace(0.5,10,10);

IV = zeros(length(TMat),length(K));
for idx = 1:1:length(TMat)
    Ttemp = TMat(idx);
    cf     =@(u) ChFBates(u, Ttemp, kappa,vBar,gamma,rho, v0, r, muJ, sigmaJ, xiP);
    valCOS    = CallPutOptionPriceCOSMthd(cf,CP,S0,r,Ttemp,K,N,L);
    figure(1)
    plot(K,valCOS)
    hold on
    for idy =1:length(K)    
        IV(idx,idy) = ImpliedVolatility(CP,valCOS(idy),K(idy),Ttemp,S0,r,0.3)*100;
    end
end
figure(2)
surf(K,TMat,IV)
ylabel('time, T')
xlabel('strike, K')
zlabel('Implied Volatility [%]')

function cf=ChFBates(u, tau, kappa,vBar,gamma,rho, v0, r, muJ, sigmaJ, xiP)
i     = complex(0,1);

% Functions D_1 and g

D_1  = sqrt(((kappa -i*rho*gamma.*u).^2+(u.^2+i*u)*gamma^2));
g    = (kappa- i*rho*gamma*u-D_1)./(kappa-i*rho*gamma*u+D_1);    

% Complex-valued functions A and C

C = (1/gamma^2)*(1-exp(-D_1*tau))./(1-g.*exp(-D_1*tau)).*(kappa-gamma*rho*i*u-D_1);
A = i*u*r*tau + kappa*vBar*tau/gamma^2 * (kappa-gamma*rho*i*u-D_1)-2*kappa*vBar/gamma^2*log((1-g.*exp(-D_1*tau))./(1-g));

% Adjustment for the Bates model

A = A - xiP*i*u*tau*(exp(muJ+1/2*sigmaJ^2)-1) + xiP*tau*(exp(i*u*muJ-1/2*sigmaJ^2*u.^2)-1);

% ChF for the Bates model

cf = exp(A + C * v0);

function value = CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L)
i = complex(0,1);


% cf   - Characteristic function, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - Interest rate (constant)
% tau  - Time to maturity
% K    - Vector of strike prices
% N    - Number of expansion terms
% L    - Size of truncation domain (typ.:L=8 or L=10)

x0 = log(S0 ./ K);   

% Truncation domain

a = 0 - L * sqrt(tau); 
b = 0 + L * sqrt(tau);

k = 0:N-1;              % Row vector, index for expansion terms
u = k * pi / (b - a);   % ChF arguments

H_k = CallPutCoefficients(CP,a,b,k);
temp    = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);     % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % matrix-vector manipulations

% Final output

value = exp(-r * tau) * K .* real(mat * temp);

% Coefficients H_k for the COS method

function H_k = CallPutCoefficients(CP,a,b,k)
    if lower(CP) == 'c' || CP == 1
        c = 0;
        d = b;
        [Chi_k,Psi_k] = Chi_Psi(a,b,c,d,k);
         if a < b && b < 0.0
            H_k = zeros([length(k),1]);
         else
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k);
         end
    elseif lower(CP) == 'p' || CP == -1
        c = a;
        d = 0.0;
        [Chi_k,Psi_k]  = Chi_Psi(a,b,c,d,k);
         H_k = 2.0 / (b - a) * (- Chi_k + Psi_k);       
    end

function [chi_k,psi_k] = Chi_Psi(a,b,c,d,k)
    psi_k        = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
    psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
    psi_k(1)     = d - c;
    
    chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2); 
    expr1 = cos(k * pi * (d - a)/(b - a)) * exp(d)  - cos(k * pi... 
                  * (c - a) / (b - a)) * exp(c);
    expr2 = k * pi / (b - a) .* sin(k * pi * ...
                        (d - a) / (b - a))   - k * pi / (b - a) .* sin(k... 
                        * pi * (c - a) / (b - a)) * exp(c);
    chi_k = chi_k .* (expr1 + expr2);
    
% Closed-form expression of European call/put option with Black-Scholes formula

function value=BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1    = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function impliedVol = ImpliedVolatility(CP,marketPrice,K,T,S_0,r,initialVol)
    func = @(sigma) (BS_Call_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice).^1.0;
    impliedVol = fzero(func,initialVol);
