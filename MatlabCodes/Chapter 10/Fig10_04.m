function SLV_Bin_Method
close all;clc;

S0 = 1;
r = 0;
x0 = log(S0);

% Heston model parameters

kappa   = 0.5;
gamma_c = 0.1;
vBar    = 0.07;
v0      = 0.07;
rho_xv  = -0.7;

% Feller condition

qparameter = 2*kappa*vBar/(gamma_c^2)-1 

NoBins  = 10;
T       = 1;
NoOfPaths = 1e5; %Number of paths per seed
NoOfSteps = 1e2*T; %Number of time steps

% -- 2DCOS 2D METHOD -- 

[Sgrid,~,EVS,~]=COSMethod2DHeston(x0,v0,rho_xv,kappa,vBar,gamma_c,T,r);

% -- Monte Carlo simulation -- 

[S,~,V] = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,r,S0,kappa,gamma_c,rho_xv,vBar,v0);
V = V(:,end);
S = S(:,end);

% -- Regression-based approach --

X1    = [ones(NoOfPaths,1) S S.^2]; Y1 = V;
beta1 = (transpose(X1)*X1)\transpose(X1)*Y1;
EV2   = X1*beta1;

matr(:,1) = S;
matr(:,2) = EV2;
matr      = sortrows(matr,1);

% -- Non-parametric pproach --

mat1 = [S getEVBinMethod(S,V,NoBins)];
mat1 = sortrows(mat1,1);

% -- Plot --

figure
hold on
p1 = scatter(S(1:NoOfPaths/10),V(1:NoOfPaths/10),'g.');
xlabel('S');ylabel('v')
xlim([0 max(S)]);ylim([0 max(V)]);
box on;grid on;
p2 = plot(Sgrid,EVS,'k--x','MarkerSize',5,'LineWidth',2);
p3 = plot(mat1(:,1),mat1(:,2),'r','LineWidth',2);
p4 = plot(mat1(:,1),matr(:,2),'b--','LineWidth',2);
legend([p1 p2 p3 p4],'Observations','COS 2D','Non-parametric','Regression-based')

figure(2)

% Contour plot

[n,c]=hist3([S,V],[30,30]);
contour(c{1},c{2},n.',6,'-b')
hold on
plot(Sgrid,EVS,'k--x','MarkerSize',5,'LineWidth',2);
xlim([0 max(S)]);ylim([0 max(V)]);
plot(mat1(:,1),mat1(:,2),'r','LineWidth',2);
grid on
xlabel('S');ylabel('v')

function phi=ChFHeston2D(u1,u2,v0,ubar,lambda,rho,eta,r,t,N1,N2)
u2=u2.';  %1xN2
beta=lambda-1i*u1*rho*eta;        %N1x1
D=sqrt(beta.^2+u1.*(1i+u1)*eta^2);  %N1x1

beta=repmat(beta,1,N2);  %N1xN2
D=repmat(D,1,N2)   ;     %N1xN2
Gtilde=(beta-D-1i*repmat(u2,N1,1)*eta^2)./(beta+D-1i*repmat(u2,N1,1)*eta^2); %N1xN2
A=lambda*ubar/eta^2*((beta-D)*t-2*log((Gtilde.*exp(-D*t)-1)./(Gtilde-1)));
B=1/eta^2*(beta-D-Gtilde.*(exp(-D*t).*(beta+D)))./(1-Gtilde.*exp(-D*t));
A=A+repmat(1i*u1,1,N2)*r*t;
phi=exp(A).*exp(v0*B);

function ksivector=ksi(z1, z2, N, a, b)

% Input arguments:  x1 = Lower bound integral psi function
%                   x2 = Upper bound integral psi function 
%                   N  = Number of terms series expansion
%                   a
%                   b           
% Output arguments: ksivector = ksi function for k=0:1:N-1

k=1:N-1;
ksitemp=-((-b + a) * cos(pi * k * (-z1 + a) / (-b + a)) ...
    + (b - a) * cos(pi * k * (-z2 + a) / (-b + a)) + ...
    pi * k .* (-z1 * sin(pi * k * (-z1 + a) / (-b + a)) ...
    + sin(pi * k * (-z2 + a) / (-b + a)) * z2)) * (-b + a) ./ (k.^2) / pi ^ 2;
ksi0=z2 ^ 2 /2 - z1 ^ 2 / 2;
ksivector=[ksi0 ksitemp];

function val = getEVBinMethod(S,v,NoBins)

if NoBins ~= 1
    mat     = sortrows([S v],1);
    BinSize = floor(length(mat)/NoBins);
    
    first = [mean(mat(1:BinSize,2)) mat(1,1) mat(BinSize,1)];
    last  = [mean(mat((NoBins-1)*BinSize:NoBins*BinSize,2)) mat((NoBins-1)*BinSize,1) mat(NoBins*BinSize,1)];
    val   = mean(mat(1:BinSize,2))*(S>=first(2) & S<first(3));
    
    for i = 2 : NoBins-1
        val = val + mean(mat((i-1)*BinSize:i*BinSize,2))*(S>=mat((i-1)*BinSize,1) & S<mat(i*BinSize,1));
    end
    
    val  = val + last(1)*(S>=last(2) & S<=last(3));
    val(val==0) = last(1);
else
    val = mean(v)*ones(length(S),1);
end

function [S,v,EVS,f]=COSMethod2DHeston(x0,v0,rho_xv,kappa,vBar,gamma_c,T,r)

% Interval [a1,b1], choice of a2 and b2.

L     = 12;
c1    = r*T+(1-exp(-kappa*T))*(vBar-v0)/(2*kappa)-0.5*vBar*T;
c2alt = vBar*(1+gamma_c)*T;
a1    = x0+c1-L*abs(sqrt(c2alt));
b1    = x0+c1+L*abs(sqrt(c2alt));
a2    = 0;
b2    = 0.5;

% Values of N1 and N2, the number of terms in the series summations.

N1=1e3; N2=1e3;

k1=0:N1-1; k2=0:N2-1;
omega1=k1'*pi/(b1-a1);  %N1x1
omega2=k2'*pi/(b2-a2); %N2x1

% Coefficienten of density function F_{k1,k2} (with characteristic function)

phiplus=ChFHeston2D(omega1,omega2,v0,vBar,kappa,rho_xv,gamma_c,r,T,N1,N2)...
    .*exp(1i*repmat(omega1,1,N2)*x0);  %N1xN2
Fplus=real(phiplus.*exp(-1i*a1*repmat(omega1,1,N2)-1i*a2*repmat(omega2',N1,1)));
phimin=ChFHeston2D(omega1,-omega2,v0,vBar,kappa,rho_xv,gamma_c,r,T,N1,N2)...
    .*exp(1i*repmat(omega1,1,N2)*x0);  %N1xN2
Fmin=real(phimin.*exp(-1i*a1*repmat(omega1,1,N2)-1i*a2*repmat(-omega2',N1,1)));
Fplus(1,1)=1;Fmin(1,1)=1;
F=1/2*(Fplus+Fmin);  %N1xN2
F(:,1)=0.5*F(:,1);  F(1,:)=0.5*F(1,:);

% Grids

xgrid=a1:(b1-a1)/150:b1;
ygrid=a2:(b2-a2)/100:b2;
cosx=cos(omega1*(xgrid-a1)); %N1 x length(xgrid)
cosy=cos(omega2*(ygrid-a2)).'; %length(xgrid) x N2
S = exp(xgrid');
v = ygrid';

% 1D density function

Fx=Fplus(:,1);  Fx(1)=0.5*Fx(1);

% f_x(X_T|x_0)

fx=2/(b1-a1)*cos((xgrid'-a1)*omega1')*Fx;

% 2D-density function %f_{x,nu}(X_T,nu_T|x_0,nu_0)

f = cosx'*F*cosy';
f = 2/(b1-a1)*2/(b2-a2)*f;

% Conditional expectation E[nu_T|X_T]

EVS  =2/(b1-a1)*2/(b2-a2)*(1./fx).*(cosx'*F*ksi(a2,b2,N2,a2,b2)');

function [S,time,V] = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,r,S_0,kappa,gamma,rho,vbar,v0)

% Define initial value

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
    fprintf('progress is %2.0f %%',i/NoOfSteps*100)
    fprintf('\n')
end
S = exp(X);

function sample = CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s)
c        = gamma^2/(4*kappa)*(1-exp(-kappa*(t-s)));
d        = 4*kappa*vbar/(gamma^2);
kappaBar = 4*kappa*exp(-kappa*(t-s))/(gamma^2*(1-exp(-kappa*(t-s))))*v_s;
sample   = c * random('ncx2',d,kappaBar,[NoOfPaths,1]);
