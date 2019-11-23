function SZHW_ImpliedVolatilities
clc;clf;close all;

% HW model parameter settings

lambda = 0.425;
eta    = 0.1;
S0     = 100;
T      = 5.0;
    
% The SZHW model

sigma0  = 0.1;
gamma   = 0.11; 
Rrsigma = 0.32;
Rxsigma = -0.42;
Rxr     = 0.3;
kappa   = 0.4;
sigmabar= 0.05;

% Strike price range

K  = linspace(40,200,20)'; 
CP = 'c';

% We define a ZCB curve (obtained from the market)

P0T = @(T) exp(-0.025*T); 
frwdStock = S0 / P0T(T); 

% Settings for the COS method

N = 2000;
L = 10;

% Effect of gamma

gammaV = [0.1, 0.2, 0.3, 0.4];
ivM = zeros(length(K),length(gammaV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(gammaV)
    gammaTemp = gammaV(i);

    % Compute ChF for the SZHW model

    cf = @(u)ChFSZHW(u,P0T,T,kappa, sigmabar,gammaTemp,lambda,eta,Rxsigma,Rxr,Rrsigma,sigma0);

    % The COS method

    valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
    valCOSFrwd = valCOS / P0T(T);     
    for j = 1:length(K)
        ivM(j,i) = ImpliedVolatilityBlack76(CP,valCOSFrwd(j),K(j),T,frwdStock,0.3)*100;
    end
    argLegend{idx} = sprintf('gamma=%.1f',gammaTemp);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'Effect of \gamma on implied volatility')

% Effect of kappa

kappaV = [0.05, 0.2, 0.3, 0.4];
ivM = zeros(length(K),length(kappaV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(kappaV)
    kappaTemp = kappaV(i);

    % Compute ChF for the SZHW model

    cf = @(u)ChFSZHW(u,P0T,T,kappaTemp, sigmabar,gamma,lambda,eta,Rxsigma,Rxr,Rrsigma,sigma0);

    % The COS method

    valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
    valCOSFrwd = valCOS / P0T(T);     
    for j = 1:length(K)
        ivM(j,i) = ImpliedVolatilityBlack76(CP,valCOSFrwd(j),K(j),T,frwdStock,0.3)*100;
    end
    argLegend{idx} = sprintf('kappa=%.2f',kappaTemp);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'Effect of \kappa on implied volatility')

% Effect of rhoxSigma

RxsigmaV = [-0.75, -0.25, 0.25, 0.75];
ivM = zeros(length(K),length(RxsigmaV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(RxsigmaV)
    RxsigmaTemp = RxsigmaV(i);

    % Compute ChF for the SZHW model

    cf = @(u)ChFSZHW(u,P0T,T,kappa,sigmabar,gamma,lambda,eta,RxsigmaTemp,Rxr,Rrsigma,sigma0);

    % The COS method

    valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
    valCOSFrwd = valCOS / P0T(T);     
    for j = 1:length(K)
        ivM(j,i) = ImpliedVolatilityBlack76(CP,valCOSFrwd(j),K(j),T,frwdStock,0.3)*100;
    end
    argLegend{idx} = sprintf('rho x,sigma=%.2f',RxsigmaTemp);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'Effect of \rho_{x,\sigma} on implied volatility')

% Effect of sigmabar

sigmaBarV = [0.1, 0.2, 0.3, 0.4];
ivM = zeros(length(K),length(sigmaBarV));
argLegend = cell(4,1);
idx = 1;
for i=1:length(sigmaBarV)
    sigmaBarTemp = sigmaBarV(i);

    % Compute ChF for the SZHW model

    cf = @(u)ChFSZHW(u,P0T,T,kappa,sigmaBarTemp,gamma,lambda,eta,Rxsigma,Rxr,Rrsigma,sigma0);

    % The COS method

    valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
    valCOSFrwd = valCOS / P0T(T);     
    for j = 1:length(K)
        ivM(j,i) = ImpliedVolatilityBlack76(CP,valCOSFrwd(j),K(j),T,frwdStock,0.3)*100;
    end
    argLegend{idx} = sprintf('sigma Bar =%.2f',sigmaBarTemp);
    idx = idx + 1;
end
MakeFigure(K, ivM,argLegend,'Effect of \bar{\sigma} on implied volatility')



function value = ChFSZHW(u,P0T,tau,kappa, sigmabar,gamma,lambda,eta,Rxsigma,Rxr,Rrsigma,sigma0)
v_D = D(u,tau,kappa,Rxsigma,gamma);
v_E = E(u,tau,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar);
v_A = A(u,tau,P0T,eta,lambda,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar);

% Initial variance 

v0     =sigma0^2;

% Characteristic function of the SZHW model

value = exp(v0*v_D + sigma0*v_E + v_A);
        
function value=C(u,tau,lambda)
    i     = complex(0,1);
    value = 1/lambda*(i*u-1).*(1-exp(-lambda*tau)); 

function value=D(u,tau,kappa,Rxsigma,gamma)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d=sqrt(a_1.^2-4*a_0.*a_2);
    g=(-a_1-d)./(-a_1+d);    
value=(-a_1-d)./(2*a_2*(1-g.*exp(-d*tau))).*(1-exp(-d*tau));

function value=E(u,tau,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d  =sqrt(a_1.^2-4*a_0.*a_2);
    g =(-a_1-d)./(-a_1+d);    
    
    c_1=gamma*Rxsigma*i*u-kappa-1/2*(a_1+d);
    f_1=1./c_1.*(1-exp(-c_1*tau))+1./(c_1+d).*(exp(-(c_1+d)*tau)-1);
    f_2=1./c_1.*(1-exp(-c_1*tau))+1./(c_1+lambda).*(exp(-(c_1+lambda)*tau)-1);
    f_3=(exp(-(c_1+d)*tau)-1)./(c_1+d)+(1-exp(-(c_1+d+lambda)*tau))./(c_1+d+lambda);
    f_4=1./c_1-1./(c_1+d)-1./(c_1+lambda)+1./(c_1+d+lambda);
    f_5=exp(-(c_1+d+lambda).*tau).*(exp(lambda*tau).*(1./(c_1+d)-exp(d*tau)./c_1)+exp(d*tau)./(c_1+lambda)-1./(c_1+d+lambda)); 

    I_1=kappa*sigmabar./a_2.*(-a_1-d).*f_1;
    I_2=eta*Rxr*i*u.*(i*u-1)./lambda.*(f_2+g.*f_3);
    I_3=-Rrsigma*eta*gamma./(lambda.*a_2).*(a_1+d).*(i*u-1).*(f_4+f_5);
    value=exp(c_1*tau).*1./(1-g.*exp(-d*tau)).*(I_1+I_2+I_3);
    
function value=A(u,tau,P0T,eta,lambda,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d  =sqrt(a_1.^2-4*a_0.*a_2);
    g =(-a_1-d)./(-a_1+d); 
    f_6=eta^2/(4*lambda^3)*(i+u).^2*(3+exp(-2*lambda*tau)-4*exp(-lambda*tau)-2*lambda*tau);
    A_1=1/4*((-a_1-d)*tau-2*log((1-g.*exp(-d*tau))./(1-g)))+f_6;
   
    % Integration within the function A(u,tau)

    value=zeros(1,length(u));   
    N=500;
    arg=linspace(0,tau,N);

    % Solve the integral for A

    for k=1:length(u)
       E_val=E(u(k),arg,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar);
       C_val=C(u(k),arg,lambda);
       f=(kappa*sigmabar+1/2*gamma^2*E_val+gamma*eta*Rrsigma*C_val).*E_val;
       value(1,k)=trapz(arg,f);
    end
    value = value + A_1;   
    
% Correction for the interest rate term structure    

help = eta^2/(2*lambda^2)*(tau+2/lambda*(exp(-lambda*tau)-1)-1/(2*lambda)*(exp(-2*lambda*tau)-1));
correction = (i*u-1).*(log(1/P0T(tau))+help);
value = value + correction;

function cfV = ChFBSHW(u, T, P0T, lambd, eta, rho, sigma)

% Time step needed 

dt = 0.0001;

% Complex number

i = complex(0.0,1.0);
f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
   
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);  
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));  
C = @(u,tau)1.0/lambd*(i*u-1.0)*(1.0-exp(-lambd*tau));
 
 % Define a grid for the numerical integration of function theta

 zGrid = linspace(0.0,T,2500);
 term1 = @(u) 0.5*sigma*sigma *i*u*(i*u-1.0)*T;
 term2 = @(u) i*u*rho*sigma*eta/lambd*(i*u-1.0)*(T+1.0/lambd *(exp(-lambd*T)-1.0));
 term3 = @(u) eta*eta/(4.0*power(lambd,3.0))*power(i+u,2.0)*(3.0+exp(-2.0*lambd*T)-4.0*exp(-lambd*T)-2.0*lambd*T);
 term4 = @(u)  lambd*trapz(zGrid,theta(T-zGrid).*C(u,zGrid));
 A= @(u) term1(u) + term2(u) + term3(u) + term4(u);
 
 % We don't include B(u)*x0 term as it is included in the COS method

 cf = @(u)exp(A(u) + C(u,T)*r0 );
 
 % Iterate over all u and collect the ChF, iteration is necessary due to the integration in term4

 cfV = zeros(1,length(u));
 idx = 1;
 for ui=u
    cfV(idx)=cf(ui);
    idx = idx +1;
 end

function value = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,tau,K,N,L,P0T)
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

k = 0:N-1;             % Row vector, index for expansion terms
u = k * pi / (b - a);  % ChF arguments

H_k = CallPutCoefficients('P',a,b,k);
temp = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

% Final output

value = K .* real(mat * temp);

% Use the put-call parity to determine call prices (if needed)

if lower(CP) == 'c' || CP == 1
 value = value + S0 - K * P0T; 
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
 psi_k  = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
 psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
 psi_k(1)  = d - c;
 
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

d1 = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2 = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
 value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
 value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function impliedVol = ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock,initialVol)
func = @(sigma) (BS_Call_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice).^1.0;
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
%  'LineWidth',1,...
%  'Color',[0 0 0]);

plot1 = plot(X1,YMatrix1,'Parent',axes1,...
 'LineWidth',1.5);
set(plot1(1),'Marker','diamond','DisplayName',argLegend{1});
set(plot1(2),'Marker','square','LineStyle','-.',...
 'DisplayName',argLegend{2});
set(plot1(3),'Marker','o','LineStyle','-.','DisplayName',argLegend{3});
set(plot1(4),'DisplayName',argLegend{4});

% Create xlabel

xlabel({'Strike, K'});

% Create ylabel

ylabel({'implied volatility [%]'});

% Create title

title(titleIn);

% Create legend

legend1 = legend(axes1,'show');
set(legend1,'Color',[1 1 1]);
