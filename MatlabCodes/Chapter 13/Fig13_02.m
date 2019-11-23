function SZHW_MonteCarlo_DiversificationProduct
clc;close all;

% HW model parameter settings

lambda = 1.12;
eta   = 0.02;
S0    = 100.0;
    
% Fixed mean reversion parameter

kappa   = 0.5;

% Diversification product

T  = 9.0;
T1 = 10.0;
    
% We define the ZCB curve (obtained from the market)

P0T = @(T)exp(-0.033*T);

% Range of the weighting factor

omegaV=  linspace(-3,3,25);

% Monte Carlo setting

NoOfPaths =5000;
NoOfSteps = round(100*T);

%               Rxr,   gamma, sigmabar, Rsigmar, Rxsigma,  sigma0

Params=        [[0.0,  0.2,   0.167,    -0.008,  -0.850,   0.035];
                [-0.7, 0.236, 0.137,    -0.339,  -0.381,   0.084]
                [0.7,  0.211, 0.102,    -0.340,  -0.850,   0.01];];

[n]=size(Params);
argLegend = cell(n(1),1);
Mrelative=zeros(length(omegaV),n(1));
Mprice=zeros(length(omegaV),n(1));
for i =1 :n(1)
    Rxr = Params(i,1);
    gamma =Params(i,2); 
    sigmaBar =Params(i,3);
    Rsigmar =Params(i,4); 
    Rxsigma =Params(i,5); 
    sigma0 =Params(i,6); 

    argLegend{i} = sprintf('rho_{x,r}=%.2f',Rxr);
    
    % Iterate over all parameters 

    randn('seed',1)
    [S_Euler,~,M_t,R] = GeneratePathsSZHW_Euler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmaBar,kappa,gamma,lambda,eta,Rxsigma,Rxr,Rsigmar);
    value = DiversifcationPayoff(P0T,S_Euler(:,end),S0,R(:,end),M_t(:,end),T,T1,lambda,eta,omegaV);
    Mprice(:,i)=value;
    if Rxr == 0.0
        refValue = value;
    end
    Mrelative(:,i)=value./refValue*100;
end

MakeFigure(omegaV, Mprice, argLegend,'Diversification product price','\omega_d','price')
MakeFigure(omegaV, Mrelative, argLegend,'Relative correlation effect','\omega_d','Value difference in BPs')

function value = DiversifcationPayoff(P0T,S_T,S0,r_T,M_T,T,T1,lambda,eta,omegaV)
P_T_T1= HW_ZCB(lambda,eta,P0T,T,T1,r_T);
P_0_T1= P0T(T1);
   
value =zeros(length(omegaV),1);
for i= 1:length(omegaV)
       payoff = omegaV(i) * S_T/S0 + (1.0-omegaV(i)) * P_T_T1./P_0_T1;
       value(i) = mean(1./M_T.*max(payoff,0.0));
end

function value = HW_B(lambda,T1,T2)
value = 1/lambda*(exp(-lambda*(T2-T1))-1);

function value = HW_theta(lambda,eta,P0T)
bump   = 10e-4;
f_0_t  =@(t)- (log(P0T(t+bump))-log(P0T(t)))/bump;
df_dt  =@(t)(f_0_t(t+bump)-f_0_t(t))/bump;
value  =@(t)f_0_t(t)+1/lambda*df_dt(t)+eta^2/(2*lambda^2)*(1-exp(-2*lambda*t));

function value = HW_A(lambda,eta,P0T,T1,T2)
tau   = T2-T1;
B     = @(tau)HW_B(lambda,0,tau);
theta = HW_theta(lambda,eta,P0T);
value = lambda*integral(@(tau)theta(T2-tau).*B(tau),0,tau)+eta^2/(4*lambda^3)...
    * (exp(-2*lambda*tau).*(4*exp(lambda*tau)-1)-3)+eta^2/(2*lambda^2)*tau;

function value = HW_ZCB(lambda,eta,P0T,T1,T2,r0)
A_r = HW_A(lambda,eta,P0T,T1,T2);
B_r = HW_B(lambda,T1,T2);
value = exp(A_r+B_r*r0);
 
function [S,time,M_t,R] = GeneratePathsSZHW_Euler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmaBar,kappa,gamma,lambd,eta,Rxsigma,Rxr,Rsigmar)

% Time step needed 

dt = 0.0001;
f0T = @(t) - (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
    
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));

% Empty containers for Brownian paths

Wx = zeros([NoOfPaths, NoOfSteps+1]);
Wsigma = zeros([NoOfPaths, NoOfSteps+1]);
Wr = zeros([NoOfPaths, NoOfSteps+1]);

Sigma = zeros([NoOfPaths, NoOfSteps+1]);
X = zeros([NoOfPaths, NoOfSteps+1]);
R = zeros([NoOfPaths, NoOfSteps+1]);
M_t = ones([NoOfPaths,NoOfSteps+1]);
R(:,1)=r0;
Sigma(:,1)=sigma0;
X(:,1)=log(S0);

covM = [[1.0, Rxsigma,Rxr];[Rxsigma,1.0,Rsigmar]; [Rxr,Rsigmar,1.0]];
time = zeros([NoOfSteps+1,1]);
        
dt = T / NoOfSteps;
for i = 1:NoOfSteps
    Z = mvnrnd([0,0,0],covM*dt,NoOfPaths);

    % Making sure that samples from a normal have mean 0 and variance 1

    Z1= Z(:,1);
    Z2= Z(:,2);
    Z3= Z(:,3);
    if NoOfPaths > 1
        Z1 = (Z1 - mean(Z1)) / std(Z1);
        Z2 = (Z2 - mean(Z2)) / std(Z2);
        Z3 = (Z3 - mean(Z3)) / std(Z3);
    end
    
    Wx(:,i+1)  = Wx(:,i)  + sqrt(dt)*Z1;
    Wsigma(:,i+1)  = Wsigma(:,i)  + sqrt(dt)*Z2;
    Wr(:,i+1) = Wr(:,i) + sqrt(dt)*Z3;
    
    R(:,i+1) = R(:,i) + lambd*(theta(time(i)) - R(:,i)) * dt + eta* (Wr(:,i+1)-Wr(:,i));
    M_t(:,i+1) = M_t(:,i) .* exp(0.5*(R(:,i+1) + R(:,i))*dt);
    
    % The volatility process

    Sigma(:,i+1) = Sigma(:,i) + kappa*(sigmaBar-Sigma(:,i))*dt+  gamma*(Wsigma(:,i+1)-Wsigma(:,i));   
     
    % Simulation of the log-stock process

    X(:,i+1) = X(:,i) + (R(:,i) - 0.5*Sigma(:,i).^2)*dt + Sigma(:,i).*(Wx(:,i+1)-Wx(:,i));       
        
    % Moment matching component, i.e. ensure that E(S(T)/M(T))= S0

    a = S0 / mean(exp(X(:,i+1))./M_t(:,i+1));
    X(:,i+1) = X(:,i+1) + log(a);
    time(i+1) = time(i) +dt;
end

    % Compute exponent

    S = exp(X);

function figure1= MakeFigure(X1, YMatrix1, argLegend,titleIn,xlabelin,ylabelin)

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

% Create xlabel

xlabel(xlabelin);

% Create ylabel

ylabel(ylabelin);

% Create title

title(titleIn);

% Create legend

legend1 = legend(axes1,'show');
set(legend1,'Color',[1 1 1]);

