function PayoffConvergence
close all;clc
randn('seed',3) %6

% Define two payoff functions

g1 = @(x)1*(x>0);
g2 = @(x)normcdf(x);

% Maturity time

T = 1;

% Range of paths

NoOfPathsV=100:100:10000;
i=1;

% Storage of the error

error1 = zeros([length(NoOfPathsV),1]);
error2 = zeros([length(NoOfPathsV),1]);

idx = 1;
for NoOfPaths=NoOfPathsV
    Z  =random('normal',0,1,[NoOfPaths,1]);
    Z  =(Z-mean(Z))/std(Z);
    W1 =sqrt(T).*Z;
    
    % Expectation

    val1 = mean(g1(W1));
    val2 = mean(g2(W1));
    
    % For both cases the exact solution is equal to 0.5

    error1(idx)=val1 - 0.5;
    error2(idx)=val2 - 0.5;
    idx=idx+1;
end

figure(1)
x=-5:0.1:5;
plot(x,g1(x),'.--','linewidth',1.5);
hold on
plot(x,g2(x),'-r','linewidth',1.5);
grid on;
xlabel('x');
ylabel('g_i(x)');
leg=legend('$$g_1(x)$$','$$g_2(x)$$');
set(leg,'Interpreter','latex');
tit=title('Payoffs $$g_1(x)$$ and $$g_2(x)$$');
set(tit,'Interpreter','latex');
%axis([-5.5,5.5,-0.05,1.05]);

figure(2)
plot(NoOfPathsV,error1,'.--','linewidth',2);
hold on
plot(NoOfPathsV,error2,'-r','linewidth',2);
grid on;
xlabel('N');
ylabel('error $$\tilde{c}_N$$','Interpreter','latex');
leg=legend('error $$\tilde{c}_N^1$$','error $$\tilde{c}_N^2$$');
set(leg,'Interpreter','latex');
tit=title('errors $$\tilde{c}_N^1$$ and $$\tilde{c}_N^2$$');
set(tit,'Interpreter','latex');
