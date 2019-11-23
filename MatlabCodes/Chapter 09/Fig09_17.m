function PathwiseSens_Heston_Delta
close all;clc;

% Monte Carlo settings

NoOfPathsMax = 25000;
NoOfSteps    = 1000;
    
% Heston model parameters

gamma = 0.5;
kappa = 0.5;
vbar  = 0.04;
rho   = -0.9;
v0    = 0.04;
T     = 1.0;
S0   = 100.0;
r     = 0.1;
CP    = 'c';

% First we define a range of strike prices and check the convergence

K = S0;

% Compute ChF for the Heston model

cf = @(u) ChFHeston(u, T, kappa,vbar,gamma,rho, v0, r);

% The COS method

dS0 = 1e-5;
optValueExact1 = CallPutOptionPriceCOSMthd(cf,CP,S0-dS0,r,T,K,1000,8);
optValueExact2 = CallPutOptionPriceCOSMthd(cf,CP,S0+dS0,r,T,K,1000,8);

% Reference delta is estimated from the COS method from the central differences

hestonDeltaExact = (optValueExact2-optValueExact1)/(2*dS0);
   
NoOfPathsV = round(linspace(5,NoOfPathsMax,25));
deltaPathWiseV = zeros([length(NoOfPathsV),1]);

idx = 1;
for nPaths = NoOfPathsV
     fprintf('Running simulation with %.f paths',nPaths);
     fprintf('\n')
     randn('seed',1);
     [S,~] = GeneratePathsHestonAES(nPaths,NoOfSteps,T,r,S0,kappa,gamma,rho,vbar,v0);
     
     % Delta- pathwise
     delta_pathwise = PathwiseDelta(S0,S,K,r,T);
     deltaPathWiseV(idx)= delta_pathwise;
     idx = idx +1;
end
 
figure(1)
grid on; hold on
plot(NoOfPathsV,deltaPathWiseV,'-or','linewidth',1.5)
plot(NoOfPathsV,hestonDeltaExact*ones([length(NoOfPathsV),1]),'linewidth',1.5)
xlabel('number of paths')
ylabel('Delta')
title('Convergence of pathwise delta w.r.t number of paths')
legend('pathwise est','exact')
ylim([hestonDeltaExact*0.8,hestonDeltaExact*1.2])

function delta= PathwiseDelta(S0,S,K,r,T)
    temp1 = S(:,end)>K;
    delta =exp(-r*T)*mean(S(:,end)/S0.*temp1); 

function [S,time] = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,r,S_0,kappa,gamma,rho,vbar,v0)

% Define initial values

V=zeros(NoOfPaths,NoOfSteps);
X=zeros(NoOfPaths,NoOfSteps);
V(:,1) = v0;
X(:,1) = log(S_0);

% Random noise

Z1=random('normal',0,1,[NoOfPaths,NoOfSteps]);
W1=zeros([NoOfPaths,NoOfSteps]);

dt = T / NoOfSteps;
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps
    if NoOfPaths>1
        Z1(:,i)   = (Z1(:,i) - mean(Z1(:,i))) / std(Z1(:,i));
    end

    % Browninan motions

    W1(:,i+1)  = W1(:,i) + sqrt(dt).*Z1(:,i);
   
    % Heston paths

    V(:,i+1) = CIR_Sample(NoOfPaths,kappa,gamma,vbar,0,dt,V(:,i));
    k0 = (r -rho/gamma*kappa*vbar)*dt;
    k1 = (rho*kappa/gamma -0.5)*dt - rho/gamma;
    k2 = rho / gamma;
    X(:,i+1) = X(:,i) + k0 + k1*V(:,i) + k2 *V(:,i+1) + sqrt((1.0-rho^2)*V(:,i)).*(W1(:,i+1)-W1(:,i));
   
    time(i+1) = time(i) + dt;
end
S = exp(X);

function sample = CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
sample   = c * random('ncx2',d,kappaBar,[NoOfPaths,1]);

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
