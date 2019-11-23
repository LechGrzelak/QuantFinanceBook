function CGMY_ImpliedVolatility
clc;clf;close all;

% Characteristic function parameters 

S0    = 100;
r     = 0.1;
sigma = 0.2;
C     = 1.0;
G     = 1.0;
M     = 5.0;
Y     = 0.5;
t     = 1.0;
CP    = 'c';

% Range of strike prices

K      = linspace(40,180,10)'; 
N      = 500;
L      = 12;

% Effect of C

CV = [0.1, 0.2, 0.5, 1.0];
ivCM = zeros(length(K),length(CV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(CV)
    Ctemp = CV(i);

    % Compute ChF for the CGMY model

    cf = ChFCGMY(r,t,Ctemp,G,M,Y,sigma);

    % The COS method

    callPrice = CallPutOptionPriceCOSMthd(cf,CP,S0,r,t,K,N,L);
    for j = 1:length(K)
        strike = K(j);
        call   = callPrice(j);
        ivCM(j,i) = ImpliedVolatility(CP,call,strike,t,S0,r,0.3)*100;
    end
    argLegend{idx} = sprintf('C=%.1f',Ctemp);
    idx = idx + 1;
end
MakeFigure(K, ivCM,argLegend,'Effect of C on implied volatility')

% Effect of G

GV = [0.5, 1.0, 2.0, 3.0];
ivGM = zeros(length(K),length(GV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(GV)
    Gtemp = GV(i);

    % Compute ChF for the CGMY model

    cf = ChFCGMY(r,t,C,Gtemp,M,Y,sigma);

    % The COS method

    callPrice = CallPutOptionPriceCOSMthd(cf,CP,S0,r,t,K,N,L);
    for j = 1:length(K)
        strike = K(j);
        call   = callPrice(j);
        ivGM(j,i) = ImpliedVolatility(CP,call,strike,t,S0,r,0.3)*100;
    end
    argLegend{idx} = sprintf('G=%.1f',Gtemp);
    idx = idx + 1;
end
MakeFigure(K, ivGM,argLegend,'Effect of G on implied volatility')

% Effect of M

MV = [2.0, 3.0, 5.0, 10.0];
ivMM = zeros(length(K),length(MV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(MV)
    Mtemp = MV(i);

    % Compute ChF for the CGMY model

    cf = ChFCGMY(r,t,C,G,Mtemp,Y,sigma);

    % The COS method

    callPrice = CallPutOptionPriceCOSMthd(cf,CP,S0,r,t,K,N,L);
    for j = 1:length(K)
        strike = K(j);
        call   = callPrice(j);
        ivMM(j,i) = ImpliedVolatility(CP,call,strike,t,S0,r,0.3)*100;
    end
    argLegend{idx} = sprintf('M=%.1f',Mtemp);
    idx = idx + 1;
end
MakeFigure(K, ivMM,argLegend,'Effect of M on implied volatility')

% Effect of Y

YV = [0.2, 0.4, 0.6, 0.9];
ivYM = zeros(length(K),length(YV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(YV)
    Ytemp = YV(i);

    % Compute ChF for the CGMY model

    cf = ChFCGMY(r,t,C,G,M,Ytemp,sigma);

    % The COS method

    callPrice = CallPutOptionPriceCOSMthd(cf,CP,S0,r,t,K,N,L);
    for j = 1:length(K)
        strike = K(j);
        call   = callPrice(j);
        ivYM(j,i) = ImpliedVolatility(CP,call,strike,t,S0,r,0.3)*100;
    end
    argLegend{idx} = sprintf('Y=%.1f',Ytemp);
    idx = idx + 1;
end
MakeFigure(K, ivYM,argLegend,'Effect of Y on implied volatility')

function cf = ChFCGMY(r,tau,C,G,M,Y,sigma)
i = complex(0,1);
varPhi = @(u) exp(tau * C *gamma(-Y)*(( M-i*u).^Y - M^Y + (G+i*u).^Y - G^Y));
omega  = -1/tau * log(varPhi(-i));
cf     = @(u) varPhi(u) .* exp(i*u* (r+ omega -0.5*sigma^2)*tau - 0.5*sigma^2 *u .*u *tau);

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

H_k = CallPutCoefficients('P',a,b,k);
temp    = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

% Final output

value = exp(-r * tau) * K .* real(mat * temp);

% Use the put-call parity to determine call prices (if needed)

if lower(CP) == 'c' || CP == 1
    value = value + S0 - K*exp(-r*tau);    
end

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
    
function MakeFigure(X1, YMatrix1, argLegend,titleIn)

%CREATEFIGURE(X1,YMATRIX1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 16-Jan-2012 15:26:40

% Create figure

figure1 = figure('InvertHardcopy','off',...
    'Colormap',[0.061875 0.061875 0.061875;0.06875 0.06875 0.06875;0.075625 0.075625 0.075625;0.0825 0.0825 0.0825;0.089375 0.089375 0.089375;0.09625 0.09625 0.09625;0.103125 0.103125 0.103125;0.11 0.11 0.11;0.146875 0.146875 0.146875;0.18375 0.18375 0.18375;0.220625 0.220625 0.220625;0.2575 0.2575 0.2575;0.294375 0.294375 0.294375;0.33125 0.33125 0.33125;0.368125 0.368125 0.368125;0.405 0.405 0.405;0.441875 0.441875 0.441875;0.47875 0.47875 0.47875;0.515625 0.515625 0.515625;0.5525 0.5525 0.5525;0.589375 0.589375 0.589375;0.62625 0.62625 0.62625;0.663125 0.663125 0.663125;0.7 0.7 0.7;0.711875 0.711875 0.711875;0.72375 0.72375 0.72375;0.735625 0.735625 0.735625;0.7475 0.7475 0.7475;0.759375 0.759375 0.759375;0.77125 0.77125 0.77125;0.783125 0.783125 0.783125;0.795 0.795 0.795;0.806875 0.806875 0.806875;0.81875 0.81875 0.81875;0.830625 0.830625 0.830625;0.8425 0.8425 0.8425;0.854375 0.854375 0.854375;0.86625 0.86625 0.86625;0.878125 0.878125 0.878125;0.89 0.89 0.89;0.853125 0.853125 0.853125;0.81625 0.81625 0.81625;0.779375 0.779375 0.779375;0.7425 0.7425 0.7425;0.705625 0.705625 0.705625;0.66875 0.66875 0.66875;0.631875 0.631875 0.631875;0.595 0.595 0.595;0.558125 0.558125 0.558125;0.52125 0.52125 0.52125;0.484375 0.484375 0.484375;0.4475 0.4475 0.4475;0.410625 0.410625 0.410625;0.37375 0.37375 0.37375;0.336875 0.336875 0.336875;0.3 0.3 0.3;0.28125 0.28125 0.28125;0.2625 0.2625 0.2625;0.24375 0.24375 0.24375;0.225 0.225 0.225;0.20625 0.20625 0.20625;0.1875 0.1875 0.1875;0.16875 0.16875 0.16875;0.15 0.15 0.15],...
    'Color',[1 1 1]);

% Create axes

%axes1 = axes('Parent',figure1,'Color',[1 1 1]);
axes1 = axes('Parent',figure1);
grid on

% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[45 160]);
% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes1,[19 26]);
% Uncomment the following line to preserve the Z-limits of the axes
% zlim(axes1,[-1 1]);

box(axes1,'on');
hold(axes1,'all');

% Create multiple lines using matrix input to plot
% plot1 = plot(X1,YMatrix1,'Parent',axes1,'MarkerEdgeColor',[0 0 0],...
%     'LineWidth',1,...
%     'Color',[0 0 0]);

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
