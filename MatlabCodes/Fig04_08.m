function SABR_ImpliedVol
close all;clc;clf;

% Global model parameters

beta  = 0.5;
alpha = 0.3;
rho   = -0.5;
nu    = 0.4;
T     = 1;
f_0   = 1;

% Figure 1, effect of beta on the implied volatility

figure(1)
hold on
str={};
ind =0;
for Beta=linspace(0.1,1,5)
    ind=ind+1;
    K=0.1:0.1:3;
    iv_h= HaganImpliedVolatility(K,T,f_0,alpha,Beta,rho,nu);
    plot(K,iv_h*100,'-b','linewidth',1.5,'color',[0 0.45 0.74])
    str{ind}=strcat('$\beta = $',num2str(Beta));
end
grid on;
xlabel('K')
ylabel('Implied volatilities [%]')
tit=title('Implied Volatilities, effect of $$\beta$$');
leg= legend(str);
set(leg,'Interpreter','latex')
set(tit,'Interpreter','latex')

% Figure 2, effect of alpha on the implied volatility

figure(2)
hold on
str={};
ind =0;
for Alpha=linspace(0.1,0.5,5)
    ind=ind+1;
    K=0.1:0.1:3;
    iv_h= HaganImpliedVolatility(K,T,f_0,Alpha,beta,rho,nu);
    plot(K,iv_h*100,'-b','linewidth',1.5,'color',[0 0.45 0.74])
    str{ind}=strcat('$\alpha = $',num2str(Alpha));
end
grid on;
xlabel('K')
ylabel('Implied volatilities [%]')
tit=title('Implied Volatilities, effect of $$\alpha$$');
leg= legend(str);
set(leg,'Interpreter','latex')
set(tit,'Interpreter','latex')

% Figure 3, effect of rho on the implied volatility

figure(3)
hold on
str={};
ind =0;
for Rho=linspace(-0.9,0.9,5)
    ind=ind+1;
    K=0.1:0.1:3;
    iv_h= HaganImpliedVolatility(K,T,f_0,alpha,beta,Rho,nu);
    plot(K,iv_h*100,'-b','linewidth',1.5,'color',[0 0.45 0.74])
    str{ind}=strcat('$\rho = $',num2str(Rho));
end
grid on;
xlabel('K')
ylabel('Implied volatilities [%]')
tit=title('Implied Volatilities, effect of $$\rho$$');
leg= legend(str);
set(leg,'Interpreter','latex')
set(tit,'Interpreter','latex')

% Figure 4, effect of gamma on the implied volatility

figure(4)
hold on
str={};
ind =0;
for Gamma=linspace(0.1,0.9,5)
    ind=ind+1;
    K=0.1:0.1:3;
    iv_h= HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,Gamma);
    plot(K,iv_h*100,'-b','linewidth',1.5,'color',[0 0.45 0.74])
    str{ind}=strcat('$\gamma = $',num2str(Gamma));
end
grid on;
xlabel('K')
ylabel('Implied volatilities [%]')
tit=title('Implied Volatilities, effect of $$\gamma$$');
leg= legend(str);
set(leg,'Interpreter','latex')
set(tit,'Interpreter','latex')

% Implied volatility according to Hagan SABR formula 

function impVol = HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma)
z        = gamma/alpha*(f*K).^((1-beta)/2).*log(f./K);
x_z      = log((sqrt(1-2*rho*z+z.^2)+z-rho)/(1-rho));
A        = alpha./((f*K).^((1-beta)/2).*(1+(1-beta)^2/24*log(f./K).^2+(1-beta)^4/1920*log(f./K).^4));
B1       = 1 + ((1-beta)^2/24*alpha^2./((f*K).^(1-beta))+1/4*(rho*beta*gamma*alpha)./((f*K).^((1-beta)/2))+(2-3*rho^2)/24*gamma^2)*T;
impVol   = A.*(z./x_z).*B1;

B2 = 1 + ((1-beta)^2/24*alpha^2./(f.^(2-2*beta))+1/4*(rho*beta*gamma*alpha)./(f.^(1-beta))+(2-3*rho^2)/24*gamma^2)*T;
impVol(K==f) = alpha/(f^(1-beta))*B2;
