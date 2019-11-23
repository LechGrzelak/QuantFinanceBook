function HestonForwardStart
clc;clf;close all;

% Heston model parameters

T_Case1= [[1,3];[2,4];[3,5];[4,6]];
T_Case2= [[1,2];[1,3];[1,4];[1,5]];
CP     = 'c';
kappa = 0.6;
gamma = 0.2;
vBar  = 0.1;
rho   = -0.5;
v0    = 0.05;  
r     = 0.00;

% Range of strike prices

K      = linspace(-0.4,4.0,25)'; 

% COS method settings

N = 500;
L = 10;

% Effect 1 on the forward implied volatilities

ivM = zeros(length(K),length(T_Case1));
argLegend = cell(4,1);
idx = 1;
for i=1:length(T_Case1)
    T1 =T_Case1(i,1);
    T2 =T_Case1(i,2);

    % Compute ChF for the Heston model

    cf = @(u) ChFHestonForward(u, T1, T2, kappa,vBar,gamma,rho, v0 ,r);

    % The COS method

    optPrice = CallPutOptionPriceCOSMthd_FrwdStart(cf,CP,r,T1,T2,K,N,L);
    for j = 1:length(K)
        strike = K(j);
        callPrice     = optPrice(j);
        ivM(j,i) = ImpliedVolatilityFrwdStart(callPrice,strike,T1,T2,r,0.3)*100;
        %ImpliedVolatility(CP,call,strike,T,S0,r,0.3)*100;
    end
    argLegend{idx} = sprintf('T_1=%.0f, T_2=%.0f',T1,T2);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'')

% Effect 2 on the forward implied volatilities

ivM = zeros(length(K),length(T_Case1));
argLegend = cell(4,1);
idx = 1;
for i=1:length(T_Case2)
    T1 =T_Case2(i,1);
    T2 =T_Case2(i,2);

    % Compute ChF for the Heston model

    cf = @(u) ChFHestonForward(u, T1, T2, kappa,vBar,gamma,rho, v0 ,r);

    % The COS method

    optPrice = CallPutOptionPriceCOSMthd_FrwdStart(cf,CP,r,T1,T2,K,N,L);
    for j = 1:length(K)
        strike = K(j);
        callPrice     = optPrice(j);
        ivM(j,i) = ImpliedVolatilityFrwdStart(callPrice,strike,T1,T2,r,0.3)*100;
        %ImpliedVolatility(CP,call,strike,T,S0,r,0.3)*100;
    end
    argLegend{idx} = sprintf('T_1=%.0f, T_2=%.0f',T1,T2);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'')

function value = ChFHestonForward(u, T_1, T_2, kappa,vBar,gamma,rho, v0 ,r)
i           = complex(0,1);
tau         = T_2-T_1;
c_bar       = @(t)gamma^2/(4*kappa)*(1-exp(-kappa*t));
d_bar       = 4*kappa*vBar/gamma^2;
kappa_bar  = @(t)4*kappa*v0*exp(-kappa*t)/(gamma^2*(1-exp(-kappa*t)));

% Functions D_1 and g

D_1  = sqrt(((kappa -i*rho*gamma.*u).^2+(u.^2+i*u)*gamma^2));
g    = (kappa- i*rho*gamma*u-D_1)./(kappa-i*rho*gamma*u+D_1);    

% Complex-valued functions A and C

C = (1/gamma^2)*(1-exp(-D_1*tau))./(1-g.*exp(-D_1*tau)).*(kappa-gamma*rho*i*u-D_1);
A = i*u*r*tau + kappa*vBar*tau/gamma^2 * (kappa-gamma*rho*i*u-D_1)-2*kappa*vBar/gamma^2*log((1-g.*exp(-D_1*tau))./(1-g));

% Coefficients used in the forward ChF

a_1 = C*c_bar(T_1)*kappa_bar(T_1)./(1-2*C*c_bar(T_1));
a_2 = (1./(1-2*C*c_bar(T_1))).^(0.5*d_bar);

% Forward start characteristic function

value = exp(A + a_1) .* a_2;

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

function value = CallPutOptionPriceCOSMthd_FrwdStart(cf,CP,r,T1,T2,K,N,L)
i = complex(0,1);


% cf   - Characteristic function, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - Interest rate (constant)
% tau  - Time to maturity
% K    - Vector of strike prices
% N    - Number of expansion terms
% L    - Size of truncation domain (typ.:L=8 or L=10)

% Adjust strike price

K = K + 1.0;

% Set up tau and initial value x0

tau = T2-T1;
x0  = log(1.0 ./ K);   

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

value = exp(-r * T2) * K .* real(mat * temp);


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

% Forward start Black-Scholes option price

function value = BS_Call_Option_Price_FrwdStart(K,sigma,T1,T2,r)
K = K + 1.0;
tau = T2 - T1;
d1    = (log(1.0 ./ K) + (r + 0.5 * sigma^2)* tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
value = exp(-r*T1) * normcdf(d1) - normcdf(d2) .* K * exp(-r * T2);

function impliedVol = ImpliedVolatility(CP,marketPrice,K,T,S_0,r,initialVol)
    func = @(sigma) (BS_Call_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice).^1.0;
    impliedVol = fzero(func,initialVol);

function impliedVol = ImpliedVolatilityFrwdStart(marketPrice,K,T1,T2,r,initialVol)
    func = @(sigma) (BS_Call_Option_Price_FrwdStart(K,sigma,T1,T2,r) - marketPrice).^1.0;
    impliedVol = fzero(func,initialVol);
    
function MakeFigure(X1, YMatrix1, argLegend,titleIn)
figure1 = figure('InvertHardcopy','off',...
    'Colormap',[0.061875 0.061875 0.061875;0.06875 0.06875 0.06875;0.075625 0.075625 0.075625;0.0825 0.0825 0.0825;0.089375 0.089375 0.089375;0.09625 0.09625 0.09625;0.103125 0.103125 0.103125;0.11 0.11 0.11;0.146875 0.146875 0.146875;0.18375 0.18375 0.18375;0.220625 0.220625 0.220625;0.2575 0.2575 0.2575;0.294375 0.294375 0.294375;0.33125 0.33125 0.33125;0.368125 0.368125 0.368125;0.405 0.405 0.405;0.441875 0.441875 0.441875;0.47875 0.47875 0.47875;0.515625 0.515625 0.515625;0.5525 0.5525 0.5525;0.589375 0.589375 0.589375;0.62625 0.62625 0.62625;0.663125 0.663125 0.663125;0.7 0.7 0.7;0.711875 0.711875 0.711875;0.72375 0.72375 0.72375;0.735625 0.735625 0.735625;0.7475 0.7475 0.7475;0.759375 0.759375 0.759375;0.77125 0.77125 0.77125;0.783125 0.783125 0.783125;0.795 0.795 0.795;0.806875 0.806875 0.806875;0.81875 0.81875 0.81875;0.830625 0.830625 0.830625;0.8425 0.8425 0.8425;0.854375 0.854375 0.854375;0.86625 0.86625 0.86625;0.878125 0.878125 0.878125;0.89 0.89 0.89;0.853125 0.853125 0.853125;0.81625 0.81625 0.81625;0.779375 0.779375 0.779375;0.7425 0.7425 0.7425;0.705625 0.705625 0.705625;0.66875 0.66875 0.66875;0.631875 0.631875 0.631875;0.595 0.595 0.595;0.558125 0.558125 0.558125;0.52125 0.52125 0.52125;0.484375 0.484375 0.484375;0.4475 0.4475 0.4475;0.410625 0.410625 0.410625;0.37375 0.37375 0.37375;0.336875 0.336875 0.336875;0.3 0.3 0.3;0.28125 0.28125 0.28125;0.2625 0.2625 0.2625;0.24375 0.24375 0.24375;0.225 0.225 0.225;0.20625 0.20625 0.20625;0.1875 0.1875 0.1875;0.16875 0.16875 0.16875;0.15 0.15 0.15],...
    'Color',[1 1 1]);

% Create axes

axes1 = axes('Parent',figure1);
grid on
box(axes1,'on');
hold(axes1,'all');

plot1 = plot(X1,YMatrix1,'Parent',axes1,...
    'LineWidth',1.5);
set(plot1(1),'Marker','diamond','DisplayName',argLegend{1});
set(plot1(2),'Marker','square','LineStyle','-.',...
    'DisplayName',argLegend{2});
set(plot1(3),'Marker','o','LineStyle','-.','DisplayName',argLegend{3});
set(plot1(4),'DisplayName',argLegend{4});

% Create xlabel

xlabel({'K'});

% Create ylabel

ylabel({'implied volatility [%]'});

% Create title

title(titleIn);

% Create legend

legend1 = legend(axes1,'show');
set(legend1,'Color',[1 1 1]);

