function StoppingTime
close all;clc;
randn('seed',20);
NoOfPaths = 3;
NoOfSteps = 250;

% Random samples

Z = random('normal',0,1,[NoOfPaths,NoOfSteps]);
W = zeros(NoOfPaths,NoOfSteps);

% Maturity time

T = 25;

% Barrier value

B = 2.0;

dt = T/NoOfSteps;
for i = 1: NoOfSteps
    if NoOfPaths>1
        Z(:,i)   = (Z(:,i) - mean(Z(:,i))) / std(Z(:,i));
    end    
    W(:,i+1) = W(:,i) + sqrt(dt) * Z(:,i);
end
time = 0:dt:T;

% Plot the Brownian Motion

figure(1)
plot(time,W,'linewidth',1.5)
grid on; hold on

% Stopping time, i.e. find the time at which the path will hit the barrier

T      =NaN(NoOfPaths,1);
Tind   =NaN(NoOfPaths,1);
PathId =NaN(NoOfPaths,1);
for i =1:NoOfPaths
    ind=find(W(i,:)>=B);
    if ~isnan(ind)
        PathId(i)=i;
        T(i)=time(ind(1));
        Tind(i)=ind(1);
    end
end

a=axis;
legend('path 1','path 2','path 3')
for i=1:length(PathId)
    x=T(i);
    plot(x,2,'.k','markersize',20)
end
plot([0,a(2)],[2,2],'--k')
xlabel('time')
ylabel('W(t)')

% Plot the histogram of the hitting times

figure(2)
hi = histogram(T,25);
hi.FaceColor = [1,1,1];
grid on
xlabel('hitting time')
ylabel('count')
