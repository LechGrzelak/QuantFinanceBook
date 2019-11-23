function H1_HW_COS_vs_MC_FX
close all;
CP  = 'c';  
T   = 5.0;

% Monte Carlo settings

NoOfPaths = 5000;
NoOfSteps = round(T*50);
    
% Settings for the COS method

N = 500;
L = 8;
    
% Market parameter settings

P0Td    = @(t) exp(-0.02*t);
P0Tf    = @(t) exp(-0.05*t);
y0      = 1.35;
frwdFX  = y0*P0Tf(T)/P0Td(T);
kappa   = 0.5;
gamma   = 0.3;
vBar    = 0.1;
v0      = 0.1;
%HW model settings
lambdad  = 0.01;
lambdaf  = 0.05;
etad    = 0.007;
etaf    = 0.012;
    
% Correlations

Rxv   = -0.4;
Rxrd  = -0.15;
Rxrf  = -0.15;
Rvrd  = 0.3;
Rvrf  = 0.3;
Rrdrf = 0.25;
 
% Strike prices

K = GenerateStrikes(frwdFX,T)';

% Valuation with the COS method

cfHHW_FX = @(u)ChFH1HW_FX(u,T,gamma,Rxv,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdad,etad,lambdaf,etaf,kappa,vBar,v0);
valCOS = P0Td(T)* CallPutOptionPriceCOSMthd_StochIR(cfHHW_FX, CP, frwdFX, T, K, N, L,1.0);

% Euler simulation of the HHW-FX model

NoOfSeeds = 20;
optionValueMC = zeros([length(K),NoOfSeeds]);
idx = 1;
for seed =1:NoOfSeeds
    randn('seed',seed)
    [FX,~]               = GeneratePathsHHWFXHWEuler(NoOfPaths,NoOfSteps,T,frwdFX,v0,vBar,kappa,gamma,lambdad,lambdaf,etad,etaf,Rxv,Rxrd,Rxrf,Rvrd,Rvrf,Rrdrf);
    optValueEuler        = P0Td(T)* EUOptionPriceFromMCPathsGeneralizedFXFrwd(CP,FX(:,end),K);
    optionValueMC(:,idx) = optValueEuler;
    fprintf('Seed number %.0f out of %.0f',idx,NoOfSeeds);
    fprintf('\n')
    idx = idx +1;
end
optionValueMCFinal = mean(optionValueMC,2);

figure(1); hold on;
plot(K,optionValueMCFinal,'-b','linewidth',1.5)
plot(K,valCOS,'--r','linewidth',1.5)
xlabel('strike,K')
ylabel('Option price')
grid on
legend('Euler','COS')
title('European option prices for the HHW model')

% Implied volatilities, the COS method vs. Monte Carlo simulation

IVHHWFX_COS = zeros(length(K),1);
IVHHWFX_MC = zeros(length(K),1);
for i=1:length(K)
    valCOS_Frwd = valCOS(i) / P0Td(T);
    valMC_Frwd = optionValueMCFinal(i) / P0Td(T);
    IVHHWFX_COS(i) = ImpliedVolatilityBlack76(CP,valCOS_Frwd,K(i),T,frwdFX,0.3)*100.0;
    IVHHWFX_MC(i) = ImpliedVolatilityBlack76(CP,valMC_Frwd,K(i),T,frwdFX,0.3)*100.0;
end

figure(2)
plot(K,IVHHWFX_COS,'-r','linewidth',1.5);hold on
plot(K,IVHHWFX_MC,'ob','linewidth',1.5)
legend('IV-H1HW-COS','IV-H1HW-MC')
grid on
title('Implied volatilities for the HHW-FX model')
xlabel('strike, K')
ylabel('Implied volatility %')

function value = GenerateStrikes(frwd,Ti)
c_n = [-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5];
value =  frwd * exp(0.1 * c_n * sqrt(Ti));

% Exact expectation E(sqrt(v(t)))

function value = meanSqrtV_3(kappa,v0,vbar,gamma_)
delta = 4.0 *kappa*vbar/gamma_/gamma_;
c= @(t) 1.0/(4.0*kappa)*gamma_^2*(1.0-exp(-kappa*(t)));
kappaBar = @(t) 4.0*kappa*v0*exp(-kappa*t)./(gamma_^2*(1.0-exp(-kappa*t)));
value = @(t) sqrt(2.0*c(t)).*gamma((1.0+delta)/2.0)/gamma(delta/2.0).*hypergeom(-0.5,delta/2.0,-kappaBar(t)/2.0);

function value = C_H1HW_FX(u,tau,kappa,gamma,Rxv)
i = complex(0.0,1.0);
D1 = sqrt((kappa-gamma*Rxv*i*u).^2.0+(u.^2+i*u)*gamma^2);
g  = (kappa-gamma*Rxv*i*u-D1)./(kappa-gamma*Rxv*i*u+D1);
value  = (1.0-exp(-D1.*tau))./(gamma^2*(1.0-g.*exp(-D1.*tau))).*(kappa-gamma*Rxv*i*u-D1);

function cf= ChFH1HW_FX(u,tau,gamma,Rxv,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa,vBar,v0)
i  = complex(0.0,1.0);
C  = @(u,tau) C_H1HW_FX(u,tau,kappa,gamma,Rxv);
Bd = @(t,T) 1.0/lambdd*(exp(-lambdd*(T-t))-1.0);
Bf = @(t,T) 1.0/lambdf*(exp(-lambdf*(T-t))-1.0);
G  = meanSqrtV_3(kappa,v0,vBar,gamma);
    
zeta = @(t) (Rxrd*etad*Bd(t,tau) - Rxrf*etaf*Bf(t,tau)).*G(t) +...
    Rrdrf*etad*etaf*Bd(t,tau).*Bf(t,tau) - 0.5*(etad^2*Bd(t,tau).^2+etaf^2*Bf(t,tau).^2);
    
% Integration within the function A(u,tau)

N    = round(25*tau);
z    = linspace(0.0+1e-10,tau-1e-10,N)';
tic
temp1 =@(z1) kappa*vBar + Rvrd*gamma*etad*G(tau-z1).*Bd(tau-z1,tau);
temp2 =@(z1,u) -Rvrd*gamma*etad*G(tau-z1).*Bd(tau-z1,tau)*i*u;
temp3 =@(z1,u)  Rvrf*gamma*etaf*G(tau-z1).*Bf(tau-z1,tau)*i*u;
f = @(z1,u) (temp1(z1)+temp2(z1,u)+temp3(z1,u)).*C(u,z1);
int1=trapz(z,f(z,u)) ;
toc
 
% int1= zeros(1,length(u));
% for k = 1:length(u)
%     temp1   = kappa*vBar + Rvrd*gamma*etad*G(tau-z).*Bd(tau-z,tau);
%     temp2   = -Rvrd*gamma*etad*G(tau-z).*Bd(tau-z,tau)*i*u(k);
%     temp3   = Rvrf*gamma*etaf*G(tau-z).*Bf(tau-z,tau)*i*u(k);
%     f       = (temp1+temp2+temp3).*C(u(k),z);
%     int1(k) = trapz(z,f);
% end

int2 = (u.^2 + i*u)*trapz(z,zeta(tau-z));
    
A = int1 + int2;
cf = exp(A + v0*C(u,tau));

function [FX,time] = GeneratePathsHHWFXHWEuler(NoOfPaths,NoOfSteps,T,frwdFX,v0,vbar,kappa,gamma,lambdd,lambdf,etad,etaf,Rxv,Rxrd,Rxrf,Rvrd,Rvrf,Rrdrf)

% Empty containers for Brownian paths

Wx = zeros([NoOfPaths, NoOfSteps+1]);
Wv = zeros([NoOfPaths, NoOfSteps+1]);
Wrd = zeros([NoOfPaths, NoOfSteps+1]);
Wrf = zeros([NoOfPaths, NoOfSteps+1]);

V = zeros([NoOfPaths, NoOfSteps+1]);
FX = zeros([NoOfPaths, NoOfSteps+1]);
V(:,1)=v0;
FX(:,1)=frwdFX;

Bd = @(t,T) 1.0/lambdd*(exp(-lambdd*(T-t))-1.0);
Bf = @(t,T) 1.0/lambdf*(exp(-lambdf*(T-t))-1.0);
covM = [[1.0, Rxv,Rxrd,Rxrf];[Rxv,1.0,Rvrd,Rvrf];...
    [Rxrd,Rvrd,1.0,Rrdrf];[Rxrf,Rvrf,Rrdrf,1.0]];

time = zeros([NoOfSteps+1,1]);
   
dt = T / NoOfSteps;
for i = 1:NoOfSteps
    Z = mvnrnd([0,0,0,0],covM*dt,NoOfPaths);

    % Making sure that samples from a normal have mean 0 and variance 1

    Z1= Z(:,1);
    Z2= Z(:,2);
    Z3= Z(:,3);
    Z4= Z(:,4);
    if NoOfPaths > 1
        Z1 = (Z1 - mean(Z1)) / std(Z1);
        Z2 = (Z2 - mean(Z2)) / std(Z2);
        Z3 = (Z3 - mean(Z3)) / std(Z3);
        Z4 = (Z4 - mean(Z4)) / std(Z4);
    end
    Wx(:,i+1)  = Wx(:,i)  + sqrt(dt)*Z1;
    Wv(:,i+1)  = Wv(:,i)  + sqrt(dt)*Z2;
    Wrd(:,i+1) = Wrd(:,i) + sqrt(dt)*Z3;
    Wrf(:,i+1) = Wrf(:,i) + sqrt(dt)*Z4;
        
    % Variance process - Euler discretization

    V(:,i+1) = V(:,i) + kappa*(vbar - V(:,i))*dt ...
            + gamma*Rvrd*etad*Bd(time(i),T)*sqrt(V(:,i)) * dt...
            + gamma* sqrt(V(:,i)).* (Wv(:,i+1)-Wv(:,i));
    V(:,i+1) = max(V(:,i+1),0.0);
        
    % FX process under the forward measure

    FX(:,i+1) = FX(:,i) .*(1.0 + sqrt(V(:,i)).*(Wx(:,i+1)-Wx(:,i)) ...
              -etad*Bd(time(i),T)*(Wrd(:,i+1)-Wrd(:,i))...
              +etaf*Bf(time(i),T)*(Wrf(:,i+1)-Wrf(:,i)));      
    time(i+1) = time(i) +dt;
end
 
function optValue = EUOptionPriceFromMCPathsGeneralizedFXFrwd(CP,S,K)
optValue = zeros(length(K),1);
if lower(CP) == 'c' || lower(CP) == 1
    for i =1:length(K)
        optValue(i) = mean(max(S-K(i),0));
    end
elseif lower(CP) == 'p' || lower(CP) == -1
    for i =1:length(K)
        optValue(i) = mean(max(K(i)-S,0));
    end
end

function sample = CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
sample   = c * random('ncx2',d,kappaBar,[NoOfPaths,1]);

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

k = 0:N-1;            % Row vector, index for expansion terms
u = k * pi / (b - a); % ChF arguments

H_k = CallPutCoefficients('P',a,b,k);
temp = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);  % Multiply the first element by 1/2

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

% Closed form expression of European call/put option with Black-Scholes formula

function value=BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * tau) ./ (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end
 
function impliedVol = ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock,initialVol)
func = @(sigma) (BS_Call_Put_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice).^1.0;
impliedVol = fzero(func,initialVol);
