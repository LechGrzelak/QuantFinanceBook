function LocalVolHagan
close all;clf;

% For the SABR model we take beta =1 and rho =0 (simpilification)

beta   = 1;
rho    = 0;

% Other model parameters

volvol = 0.2;
s0     = 1;
T      = 10;
r      = 0.05;
alpha  = 0.2;
f_0    = s0*exp(r*T);

% Monte Carlo settings

NoOfPaths = 25000;
NoOfSteps = 100*T;
randn('seed',2)

% We define the market to be driven by Hagan's SABR formula
% Based on this formula we derive local volatility/variance 

sigma     = @(x,t) HaganImpliedVolatility(x,t,f_0,alpha,beta,rho,volvol);
sigma_ATM = HaganImpliedVolatility(f_0,T,f_0,alpha,beta,rho,volvol);

% Local variance based on the Hagan's SABR formula

sigmalv2  = LocalVarianceBasedOnSABR(s0,f_0,r,alpha,beta,rho,volvol);

dt        = T/NoOfSteps;
Z         = random('normal',0,1,[NoOfPaths,NoOfSteps]);
S         = zeros(NoOfPaths,NoOfSteps+1);

S(:,1)    = s0;
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps

    % This condition is necessary as for t=0 we cannot compute implied
    % volatilities

    if time(i)==0
        time(i)=0.0001;
    end
    sprintf('current time is %3.3f',time(i))

    % Standarize Normal(0,1)

    Z(:,i)=(Z(:,i)-mean(Z(:,i)))/std(Z(:,i));
    
    % Compute local volatility

    sig=real(sigmalv2(S(:,i),time(i)));

    % Because of discretizations we may encounter negative variance which
    % needs to be set to 0.

    sig=max(sig,0);
    sigmaLV = sqrt(sig);  
    
    % Stock path

    S(:,i+1)=S(:,i) .*(1 + r*dt + sigmaLV.*Z(:,i).*sqrt(dt));
    
    % We force at each time S(t)/M(t) to be a martingale

    S(:,i+1)= S(:,i+1) - mean(S(:,i+1)) + s0*exp(r*time(i));

    % Make sure that after moment matching we don't encounter negative stock values

    S(:,i+1)=max(S(:,i+1),0);

    % Adjust time

    time(i+1) = time(i) + dt;
end

% Compute implied volatility 

K=0.2:0.15:5;
IV_Hagan   = zeros(length(K),1);
CallPrices = zeros(length(K),1);
IV_MC      = zeros(length(K),1);
for i=1:length(K)
    IV_Hagan(i) =sigma(K(i),T);
    CallPrices(i)    =exp(-r*T).*mean(max(S(:,end)-K(i),0));
    IV_MC(i)    =ImpliedVolatility('c',CallPrices(i),K(i),T,s0,r,0.2); 
end

figure(1)
plot(K,IV_Hagan*100,'linewidth',1.5)
hold on
plot(K,IV_MC*100,'.r','MarkerSize',10)
title('Local Volatility model for European Options')
grid on;
xlabel('Strike, K')
ylabel('Implied volatility [%]')
plot(K,ones(length(K),1)*sigma_ATM*100,'--k','linewidth',1.5)
legend1 = legend('$$\hat\sigma(T,K)$$','$$\sigma_{imp}(T,K)$$','ATM Black-Scholes imp. vol');
set(legend1,'Interpreter','latex');

figure(2)

% Lognormal density (Black-Scholes) compared to density from generated paths

[y,x]=ksdensity(S(:,end),[0:0.01:7]);
f_S_logn =@(x)1./(sigma_ATM*x*sqrt(2*pi*T)).*exp(-(log(x./s0)-(r-0.5*sigma_ATM^2)*T).^2./(2*sigma_ATM*sigma_ATM*T));
plot(x,y,'linewidth',1.5)
hold on
plot(x,f_S_logn(x),'--r','linewidth',1.5)
xlabel('x')
ylabel('probability density function')
title('Local Volatility vs. Black Scholes')
grid on
legend('Local Volatility','Black Scholes')

% Implied volatility according to Hagan et al. SABR formula 

function impVol = HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma)
z        = gamma/alpha*(f*K).^((1-beta)/2).*log(f./K);
x_z      = log((sqrt(1-2*rho*z+z.^2)+z-rho)/(1-rho));
A        = alpha./((f*K).^((1-beta)/2).*(1+(1-beta)^2/24*log(f./K).^2+(1-beta)^4/1920*log(f./K).^4));
B1       = 1 + ((1-beta)^2/24*alpha^2./((f*K).^(1-beta))+1/4*(rho*beta*gamma*alpha)./((f*K).^((1-beta)/2))+(2-3*rho^2)/24*gamma^2)*T;
impVol   = A.*(z./x_z).*B1;

B2 = 1 + ((1-beta)^2/24*alpha^2./(f.^(2-2*beta))+1/4*(rho*beta*gamma*alpha)./(f.^(1-beta))+(2-3*rho^2)/24*gamma^2)*T;
impVol(K==f) = alpha/(f^(1-beta))*B2;

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
    
function value = LocalVarianceBasedOnSABR(s0,frwd,r,alpha,beta,rho,volvol)

% Define shock size for approximating derivatives

dt =0.001;
dx =0.001;

% Function hold for Hagan's implied volatility

sigma =@(x,t) HaganImpliedVolatility(x,t,frwd,alpha,beta,rho,volvol);

% Derivatives

dsigmadt   = @(x,t) (sigma(x,t+dt)-sigma(x,t))./dt;
dsigmadx   = @(x,t) (sigma(x+dx,t)-sigma(x-dx,t))./(2*dx);
d2sigmadx2 = @(x,t) (sigma(x+dx,t) + sigma(x-dx,t)-2*sigma(x,t))./(dx.*dx);

omega      = @(x,t) sigma(x,t).*sigma(x,t)*t;
domegadt   = @(x,t)sigma(x,t).^2 + 2*t*sigma(x,t).*dsigmadt(x,t);
domegadx   = @(x,t)2*t*sigma(x,t).*dsigmadx(x,t);
d2omegadx2 = @(x,t)2*t*(dsigmadx(x,t)).^2 + 2*t*sigma(x,t).*d2sigmadx2(x,t);

term1    = @(x,t)1+x.*domegadx(x,t).*(0.5 - log(x./(s0*exp(r*t)))./omega(x,t));
term2    = @(x,t)0.5*x.^2.*d2omegadx2(x,t);
term3    = @(x,t)0.5*x.^2.*domegadx(x,t).^2.*(-1/8-1./(2.*omega(x,t))+log(x./(s0*exp(r*t))).*log(x./(s0*exp(r*t)))./(2*omega(x,t).*omega(x,t)));

% Final expression for local variance

sigmalv2 = @(x,t)(domegadt(x,t)+r*x.*domegadx(x,t))./(term1(x,t)+term2(x,t)+term3(x,t));
value = sigmalv2;
