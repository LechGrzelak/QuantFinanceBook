function StochasticIntegrals
close all;
randn('seed',8)
NoOfPaths = 1;
NoOfSteps = 1000;

T = 1;
Z = random('normal',0,1,[NoOfPaths,NoOfSteps]);
W = zeros(NoOfPaths,NoOfSteps);
I1 = zeros(NoOfPaths,NoOfSteps);
I2 = zeros(NoOfPaths,NoOfSteps);
dt= T/NoOfSteps;
for i = 1:NoOfSteps
    if NoOfPaths >1
        Z(:,i) = (Z(:,i)- mean(Z(:,i)))/std(Z(:,i));
    end
    W(:,i+1)  = W(:,i) + sqrt(dt)*Z(:,i);
    I1(:,i+1) = I1(:,i) + W(:,i)*dt;
    I2(:,i+1) = I2(:,i) + W(:,i).*(W(:,i+1)-W(:,i));
end
time_grid = 0:dt:T;

% Plotting results and error calculation

figure(1)
plot(time_grid,W,'linewidth',1.5,'color',[0 0.45 0.74])
hold on
plot(time_grid,I1,'-r','linewidth',1.5)
plot(time_grid,I2,'k','linewidth',1.5)
grid on
xlabel('time, t')
ylabel('value')

error_1 = mean(I1(:,end)) 
error_2 = mean(I2(:,end)) - 0.5*mean(W(:,end).^2)+ 0.5*T
legend1 = legend('W(t)','$$I(t)=\int_0^tW(s)ds$$','$$I(t)=\int_0^tW(s)dW(s)$$');
set(legend1,'Interpreter','latex');
title('Integrated Brownian Motion paths')
