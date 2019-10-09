function CashOrNothing_COS_Method
clc;close all;

% Pricing of cash-or-nothing options with the COS method

i      = complex(0,1);
CP     = 'c';
L      = 6;
K      = [120]; 
tau    = 0.1;
r      = 0.05;
sigma  = 0.2;
N      = [40, 60, 80, 100, 120, 140];
S0     = 100;

% Definition of the ChH for GBM; this is an input for the COS method
% The Chf does not contain the coefficient "+iuX(t_0)" as this coefficient
% is included internally in the evaluation 
% In the book we denote this function by \varphi(u)

cf = @(u)exp((r - 1 / 2 * sigma^2) * i * u * tau - 1/2 * sigma^2 * u.^2 * tau);

% Closed-form price expression

val_Exact= BS_Cash_Or_Nothing_Price(CP,S0,K,sigma,tau,r);

% Timing results

 NoOfIterations = 100;
 errorVec = zeros(length(N),1);
 idx = 1;
for n=N
    tic
    for i = 1:NoOfIterations
        val_COS = CashOrNothingPriceCOSMthd(cf,CP,S0,r,tau,K,n,L);
    end
    errorVec(idx)=val_Exact - val_COS;
    time_elapsed = toc;
    sprintf('For N= %.0f it took %f seconds to price',n,time_elapsed/NoOfIterations)
    sprintf('For N= %.0f the error is equal to %e',n,errorVec(idx))
    idx = idx +1;
end

% Plot the results

figure(1); clf(1); 
plot(N,errorVec,'-.r');hold on
grid on;xlabel('Number of expansion terms')
ylabel('Error')
 
function value = CashOrNothingPriceCOSMthd(cf,CP,S0,r,tau,K,N,L)
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

H_k = CashOrNothingCoefficients(CP,a,b,k);
temp    = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

% Final output

value = exp(-r * tau) * K .* real(mat * temp);

% Coefficients H_k for the COS method

function H_k = CashOrNothingCoefficients(CP,a,b,k)
    if lower(CP) == 'c' || CP == 1
        c = 0;
        d = b;
        [~,Psi_k] = Chi_Psi(a,b,c,d,k);
         if a < b && b < 0.0
            H_k = zeros([length(k),1]);
         else
            H_k = 2.0 / (b - a) * Psi_k;
         end
    elseif lower(CP) == 'p' || CP == -1
        c = a;
        d = 0.0;
        [~,Psi_k]  = Chi_Psi(a,b,c,d,k);
         H_k = 2.0 / (b - a) * Psi_k;       
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
    
% Closed-form expression of cash-or-nothing option with the Black-Scholes formula

function value=BS_Cash_Or_Nothing_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1    = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =  K * exp(-r * tau).*normcdf(d2) ;
elseif lower(CP) == 'p' || lower(CP) == -1
    value = K*exp(-r*tau)*(1-normcdf(d2));
end
    
    
