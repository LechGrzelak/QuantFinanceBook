function BrownianDiscretization
clc;close all;

% Fix number of the Monte Carlo paths

NoOfPaths = 50000;

% Range for the discretization 

mVec = 2:50;
T    = 1;

% Storage of mean and variance 

meanV = zeros(length(mVec),1);
varV = zeros(length(mVec),1);

indx = 1;
for m = mVec
    t1 = 1*T/m;
    t2 = 2*T/m;
    % Condier two Brownian motions W(t1) and W(t2)
    Wt1 = sqrt(t1)*random('normal',0,1,[NoOfPaths,1]);
    Wt2 = Wt1 + sqrt(t2-t1)*random('normal',0,1,[NoOfPaths,1]);
    X = (Wt2-Wt1).^2;
    meanV(indx) = mean(X);
    varV(indx) = var(X);
    indx = indx +1;
end

figure(1)
plot(mVec,meanV,'-b');hold on
plot(mVec,varV,'--r')
hold on;
grid on
legend('E(W(t_i+1)-W(t_i))^2','Var(W(t_i+1)-W(t_i))^2')
