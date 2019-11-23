function LocalVolHeston
format long;close all;
randn('seed',8); % Random seed
NoOfPaths  =10000;

CP         ='c';
S0         =1;       % Initial stock
r          =0.06;    % Interest rate
tau        =1;       % Maturity
NoOfSteps  =100*tau;
dt         =tau/NoOfSteps; %Tme increament

% Heston model parameters

kappa   = 1.3; 
vbar    = 0.05; 
gamma   = 0.3; 
rho     = -0.3; 
v0      = 0.1;

% Define a function to calculate option prices

V   = @(T,K)OptionPriceWithCosMethodHelp(CP,T,K,S0,r,kappa,gamma,v0,vbar,rho);

% Define bump size

bump_T  = 1e-4;
bump_K  = @(T)1e-4;

% Define derivatives

dC_dT   = @(T,K) (V(T + bump_T,K) - V(T ,K)) /  bump_T;
dC_dK   = @(T,K) (V(T,K + bump_K(T)) - V(T,K - bump_K(T))) / (2 * bump_K(T));
d2C_dK2 = @(T,K) (V(T,K + bump_K(T)) + V(T,K-bump_K(T)) - 2*V(T,K)) / bump_K(T)^2;

% Local volatility

sigma2LV=@(T,K)(dC_dT(T,K) + r * K .* dC_dK(T,K))...
        ./ (0.5 .* K.^2 .* d2C_dK2(T,K));

% Prepare stock and Brownian motion

S       =zeros(NoOfPaths,1);
S(:,1)  =S0;
Z       =random('normal',0,sqrt(dt),[NoOfPaths,NoOfSteps]);

t=0;
for i=1:NoOfSteps

    % Adjust noise so that ~ N(0,sqrt(dt))

    Z(:,i)=(Z(:,i)-mean(Z(:,i)))/std(Z(:,i))*sqrt(dt);
    if t==0
                sigma_t =sqrt(sigma2LV(0.001, S));
    else
                sigma_t =sqrt(max(sigma2LV(t, S),0));         
    end
    S = S .* (1 + r * dt + sigma_t .* Z(:,i));
    
    S=(S-mean(S))+S0*exp(r*(t+dt));
    S= max(S,0);
    t = t + dt                         
end

% Calculate implied volatilities: local vol. vs. the Heston model

K          =0.5:0.1:1.7; % Strikes at which smile is determined

% Initialization of all variables (increases Matlab efficiency)

error      =zeros(length(K),1);
call_LV=error; imp_vol_LV=error; imp_vol_Heston=error;

% For each strike, determine the implied volatility

OptionValueHeston    = V(tau,K');
for j=1:length(K)

    % Local volatility

    call_LV(j)        = exp(-r * tau) * mean( max(S - K(j),0) );
    imp_vol_LV(j)     = ImpliedVolatility(CP,call_LV(j),K(j),tau,S0,r,0.2);

    % Heston

    imp_vol_Heston(j) = ImpliedVolatility(CP,OptionValueHeston(j),K(j),tau,S0,r,0.2);    

    % Error

    error(j)=imp_vol_LV(j)-imp_vol_Heston(j);  
end
figure(1)
hold on
plot(K,imp_vol_LV*100,'linewidth',1.5)
plot(K,imp_vol_Heston*100,'--r','linewidth',1.5)
xlabel('strike, K')
ylabel('Implied volatility [%]')
grid on
legend('Local volatility','Stochastic Volatility (Heston)')
xlim([min(K),max(K)])

function value = OptionPriceWithCosMethodHelp(CP,T,K,S0,r,kappa,gamma,v0,vbar,rho)
cf   = @(u) ChFHeston(u, T, kappa,vbar,gamma,rho, v0, r);

% The COS method

% Settings for the COS method

N = 500;
L = 8;
value = CallPutOptionPriceCOSMthd(cf,CP,S0,r,T,K,N,L);

function cf=ChFHeston(u, tau, kappa,vBar,gamma,rho, v0, r)
i     = complex(0,1);

% Functions D_1 and g

D_1  = sqrt(((kappa -i*rho*gamma.*u).^2+(u.^2+i*u)*gamma^2));
g    = (kappa- i*rho*gamma*u-D_1)./(kappa-i*rho*gamma*u+D_1);    

% Complex-valued functions A and C

C = (1/gamma^2)*(1-exp(-D_1*tau))./(1-g.*exp(-D_1*tau)).*(kappa-gamma*rho*i*u-D_1);
A = i*u*r*tau + kappa*vBar*tau/gamma^2 * (kappa-gamma*rho*i*u-D_1)-2*kappa*vBar/gamma^2*log((1-g.*exp(-D_1*tau))./(1-g));

% ChF for the Heston model

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
temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

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
