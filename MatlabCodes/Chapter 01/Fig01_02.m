function BivariateNormal
close all;clc;

% Number of points in each direction

n    = 30;

% Parameters

mu_1    = 0;
mu_2    = 0;
sigma_1 = 1;
sigma_2 = 0.5;
[X,Y]   = meshgrid(linspace(-3,3,n),linspace(-3,3,n));

PDF2D =@(rho) mvnpdf([X(:),Y(:)],[mu_1,mu_2],[sigma_1^2,rho*sigma_1*sigma_2;rho*sigma_1*sigma_2,sigma_2^2]);

rho1 = 0;
rho2 = -0.8; 
rho3 = 0.8; 
PDFrho1 = PDF2D(rho1);
PDFrho2 = PDF2D(rho2);
PDFrho3 = PDF2D(rho3);

figure1 = figure('PaperSize',[40.99999864 60]);
set(gcf, 'Position', [400, 500, 1000, 300])
subplot(1,3,1,'Parent',figure1)
mesh(X,Y,reshape(PDFrho1,n,n),'edgecolor', 'k')
xlabel('x')
ylabel('y')
zlabel('PDF')
axis([min(min(X)),max(max(X)),min(min(Y)),max(max(Y)),0,max(max(PDFrho1))])

subplot(1,3,2,'Parent',figure1)
mesh(X,Y,reshape(PDFrho2,n,n),'edgecolor', 'k')
xlabel('x')
ylabel('y')
zlabel('PDF')
axis([min(min(X)),max(max(X)),min(min(Y)),max(max(Y)),0,max(max(PDFrho2))])

subplot(1,3,3,'Parent',figure1)
mesh(X,Y,reshape(PDFrho3,n,n),'edgecolor', 'k')
xlabel('x')
ylabel('y')
zlabel('PDF')
axis([min(min(X)),max(max(X)),min(min(Y)),max(max(Y)),0,max(max(PDFrho3))])


