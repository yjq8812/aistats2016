function [f,g,subIsIn] = testLossFunc(loss,len)
% this script gives some example non-modular (symetric) loss function 
% given their decomposition fasion l=f+g, with different properties on f and g
% the loss function can be accessed simply by f(x)+g(x)
switch loss
    case 1
        % LossFunction 1
        % l: one step, non-submodular, f: non-negative inreasing
        f = @(x)(min(ceil(length(x)/4),length(find(x)))/ceil(length(x)/4));
        subIsIn = 1; 
        g = @(x)(max(0,length(find(x))-ceil(length(x)*3/4))/ceil(length(x)/4));
    case 2
        % LossFunction 2 
        % l: submodular, f: non-negative, non-increasing,f(all)>0
        f = @(x)(min(min(length(find(x)) , fix(length(x)/4) ), 0.9*length(x) -0.8*length(find(x)))  /ceil(length(x)/4));
        subIsIn = 0; 
        g = @(x)(max(0, length(find(x))-fix(length(x)/2) ) / ceil(length(x)/4));    
    case 3
        % LossFunction 3
        % l: one step function , f: negative decreasing(not very nice for Lovasz Hinge)
        f = @(x)(min(0,ceil(length(x)*2/3)-length(find(x)))/ceil(length(x)/3));
        subIsIn = 0; 
        g = @(x)(max(0,length(find(x))-fix(length(x)/3))/ceil(length(x)/3));
    case 4
        % LossFunction 4 with the end drop i.e. loss for disease diagnosis 
        % l: one step function , f: negative decreasing(not very nice for Lovasz Hinge)
        f = @(x)(min(0,ceil(length(x)*2/3)-length(find(x)))/ceil(length(x)/3));
        subIsIn = 0; 
        g = @(x)(max(0,0.75*(length(find(x))-fix(length(x)/3))/ceil(length(x)/3)));       
    otherwise
        error('the function is not defined')
end

% if there is a second input, this function will plot the loss in 2D
if exist('len','var')
lens = len;
figure;

for l = 1: length(lens)
    len = lens(l);
    x = zeros(len,1);
    if length(lens)>1
    subplot(ceil(length(lens)/4),4,l)
    end
    for i=1:len
        
        x(1:i) = 1;
        y_f(i) = f(x);
        y_g(i) = g(x);
        y_l(i) = f(x)+g(x);
        
    end
    
    hold on;
    grid on;
    box on;
    plot([0:len],[0,y_l],'-^r','LineWidth',4.5)
    plot([0:len],[0,y_f],'-.*b','LineWidth',4.5)
    plot([0:len],[0,y_g],'-.og','LineWidth',4.5)
        % ylim([0 2])
    hleg = legend('l','f', 'g');
    set(hleg,'FontAngle','italic','TextColor',[.3,.2,.1],'Location','NorthWest')
    set(hleg,'Box','off');
    %     title(['length of bag = ' num2str(len)],'FontSize',22,'fontWeight','bold')
    set(gca,'FontSize',22,'fontWeight','bold')
end
end
end
