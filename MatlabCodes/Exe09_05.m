function martingale
close all;clc;
randn('seed',3)
t=10;
s=5;
NoOfPaths = 1000;
NoOfSteps = 100;

% Case a) E(W(t)|F(0))= W(0)- answer is deterministic

W_t= random('normal',0,sqrt(t),[NoOfPaths,1]);

answerNum  = mean(W_t)

% Since W(0)= 0, we simply have the following

answerTheo = 0
error= answerNum - answerTheo

% Case b) E(W(t)|F(s))= W(s) - answer is stochastic

% we simulate paths W(s) until time s

W = zeros(NoOfPaths,NoOfSteps+1);
Z = random('normal',0,1,[NoOfPaths,NoOfSteps]);
dt1 = s/NoOfSteps;
for i = 1: NoOfSteps
    % this is a scaling that ensures that Z has mean 0 and variance 1
    Z(:,i) = (Z(:,i)-mean(Z(:,i)))/std(Z(:,i));
    % path simulation
    W(:,i+1) = W(:,i) + sqrt(dt1)*Z(:,i);
end

% W_s is the last column of W

W_s =W(:,end);

% For every path W(s) we perform sub-simulation until time t and calculate
% the expectation

dt2     = (t-s)/NoOfSteps;
W_t     = zeros(NoOfPaths,NoOfSteps+1);
E_W_t   = zeros(NoOfPaths,1);
for i   = 1 : NoOfPaths
    i

    % Sub-simulation from time "s" until "t"

    Z = random('normal',0,1,[NoOfPaths,NoOfSteps]);
    W_t(:,1) = W_s(i);
    for j = 1: NoOfSteps

        % This is a scaling which ensures that Z has mean 0 and variance 1

        Z(:,j) = (Z(:,j)-mean(Z(:,j)))/std(Z(:,j));

        % Path simulation

        W_t(:,j+1) = W_t(:,j) + sqrt(dt2)*Z(:,j);
    end 
    E_W_t(i) = mean(W_t(:,end));
    
    % Plotting some figures

    %      figure(1)
    %      plot(0:dt1:s,W(i,:),'-r','linewidth',1.5)
    %      hold on;
    %      plot(s:dt2:t,W_t(:,:),'-k','linewidth',1)
    %      xlabel('time')
    %      ylabel('W(t)')
    %      grid on

     W_t=zeros(NoOfPaths,NoOfSteps+1);  
end
error = max(abs(E_W_t-W_s))


stop=1;
