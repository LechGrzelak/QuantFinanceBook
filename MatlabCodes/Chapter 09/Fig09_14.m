function ImportanceSampling
clc;close all
format short

% Exact solution P(X>5)

a        = 3;
exactSol = 1-normcdf(a,0,1);

% Specification of a grid for the number of samples

gridN = 1000:100:100000; 

% Creation of empty vectors to store the output

Solution  =  zeros(length(gridN),1);
SolutionCase1  =  zeros(length(gridN),1);
SolutionCase2  =  zeros(length(gridN),1);
SolutionCase3  =  zeros(length(gridN),1);
SolutionCase4  =  zeros(length(gridN),1);

% Function holds for Radon-Nikodym derivative

L1 = @(x)normpdf(x,0,1)./unifpdf(x,0,1);
L2 = @(x)normpdf(x,0,1)./unifpdf(x,0,4);
L3 = @(x)normpdf(x,0,1)./normpdf(x,0,0.5);
L4 = @(x)normpdf(x,0,1)./normpdf(x,0,3);

% Running index;

i=1;

for N=gridN

    % Standard approach

    X = random('normal',0,1,[N,1]);
    RV = double(X>a);
    Pr=mean(RV);
    Solution(i)=Pr;
    
    % Case 1, uniform(0,1)

    Y = random('uniform',0,1,[N,1]);
    RV = double(Y>a).*L1(Y);
    Pr=mean(RV);
    SolutionCase1(i)=Pr;
    
    % Case 2, uniform(0,4)

    Y = random('uniform',0,4,[N,1]);
    RV = double(Y>a).*L2(Y);
    Pr=mean(RV);
    SolutionCase2(i)=Pr;
    
    % Case 3, normal(0,0.5)

    Y = random('normal',0,0.5,[N,1]);
    RV = double(Y>a).*L3(Y);
    Pr=mean(RV);
    SolutionCase3(i)=Pr;
    
    % Case 4, normal(0,3)

    Y = random('normal',0,3,[N,1]);
    RV = double(Y>a).*L4(Y);
    Pr=mean(RV);
    SolutionCase4(i)=Pr;
    
    i=i+1;
end
%plot(gridN,Variance,'-r','linewidth',2)
plot(gridN,Solution,'.')
hold on
plot(gridN,SolutionCase1,'.k')
plot(gridN,SolutionCase2,'.g')
plot(gridN,SolutionCase3,'.m')
plot(gridN,SolutionCase4,'.y')
plot(gridN,exactSol*ones(length(gridN),1),'-r','linewidth',2)
grid on
xlabel('Number of samples')
ylabel('Estimated value')
legend('Standard Monte Carlo','U([0,1])','U([0,4])','N(0,0.25)', 'N(0,9)','exact')
    
