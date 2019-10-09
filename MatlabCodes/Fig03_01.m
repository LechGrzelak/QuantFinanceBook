function PathsAndPayoff
clc;close all;

% Parameter settings for GBM process

S0        = 100;
NoOfPaths = 2;
NoOfSteps = 100;
r         = 0.05;
sigma     = 1;
T         = 1;
dt        = T/NoOfSteps;
X         = zeros(NoOfPaths,NoOfSteps);
X(:,1)    = log(S0);

randn('seed',43) 
Z=random('normal',0,1,[NoOfPaths,NoOfSteps]);

% Simulation of GBM process

for i=1:NoOfSteps
   X(:,i+1) = X(:,i) + (r-0.5*sigma^2)*dt + sigma* sqrt(dt)*Z(:,i);
end
S    = exp(X);
x= 0:dt:T;

%% First figure

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
%plot(x,S(:,:),'linewidth',1.5)
plot(x,S(1,:),'linewidth',1.5)
plot(x,S(2,:),'linewidth',1.5,'linestyle','-.','color',[1,0,0])
xlabel('time')
set(axes1,'XAxisLocation','origin','YAxisLocation','right');
ylabel('S(T)')
%title('Two random paths')
%set(axes1,'XTick',zeros(1,0));
%set(axes1,'YTick',zeros(1,0));
coef = 1.5;
text(-0.01,S0,'S(t_0)','HorizontalAlignment','right')
plot(0,S0,'MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 0])
plot(T,S0*coef,'MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 0])
plot(T,S(1,end),'MarkerFaceColor',[0 0.45 0.74],'MarkerEdgeColor',[0 0.45 0.74],'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0.45 0.74])
plot(T,S(2,end),'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[1 0 0])
text(T*0.96,S0*coef,'K','HorizontalAlignment','left')
axis([0,T,0,max(max(S))])
plot([T,T],[0,max(max(S))],'Linewidth',0.75,'color',[0,0,0],'LineStyle','--')
text(0.02,-25,'t_0','HorizontalAlignment','right')
text(T,-25,'T','HorizontalAlignment','right')
grid on

%% Second figure

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
%plot(x,S(:,:),'linewidth',1.5)
plot(x,S(1,:),'linewidth',1.5)
plot(x,S(2,:),'linewidth',1.5,'linestyle','-.','color',[1,0,0])
xlabel('time')
set(axes1,'XAxisLocation','origin','YAxisLocation','right');
ylabel('S(T)')
%title('Two random paths')
%set(axes1,'XTick',zeros(1,0));
%set(axes1,'YTick',zeros(1,0));
coef = 1.5;
text(-0.01,S0,'S(t_0)','HorizontalAlignment','right')
plot(0,S0,'MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 0])
plot(T,S0*coef,'MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 0])
plot(T,S(1,end),'MarkerFaceColor',[0 0.45 0.74],'MarkerEdgeColor',[0 0.45 0.74],'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0.45 0.74])
plot(T,S(2,end),'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0],'Marker','o',...
    'LineStyle','none',...
    'Color',[1 0 0])
text(T*0.96,S0*coef,'K','HorizontalAlignment','left')
axis([0,T,0,max(max(S))])

plot([T*0.8,T*0.8],[0,max(max(S))],'Linewidth',0.75,'color',[0,0,0],'LineStyle','--')
plot([T*0.6,T*0.6],[0,max(max(S))],'Linewidth',0.75,'color',[0,0,0],'LineStyle','--')
plot([T*0.4,T*0.4],[0,max(max(S))],'Linewidth',0.75,'color',[0,0,0],'LineStyle','--')
plot([T,T],[0,max(max(S))],'Linewidth',0.75,'color',[0,0,0],'LineStyle','--')
text(T*0.4,-25,'T_1','HorizontalAlignment','right')
text(T*0.6,-25,'T_2','HorizontalAlignment','right')
text(T*0.8,-25,'T_3','HorizontalAlignment','right')
text(0.02,-25,'t_0','HorizontalAlignment','right')
text(T,-25,'T','HorizontalAlignment','right')
grid on
