function [patterns, labels] = generateSyntheticData(numpattern,sizebag, percentage,draw)
%   Create a toy problem for Earlier Penalization Problem
%   in 2-D feature, patterns = [x,y]
%   numpattern  : number of patterns
%   sizebag     : size of each bag
%   percentage  : the precentage of earlier ones we care

if(~exist('sizebag','var'))
   sizebag = 15;
end

if(~exist('percentage','var'))
    percentage = 3/sizebag ;
end
if(~exist('draw','var'))
   draw = 0;
end


patterns = cell(1,numpattern) ;
labels = cell(1,numpattern) ;


dim = 2;
numclass = 2;

positive_early = randn(numpattern/numclass,sizebag*percentage,dim);

positive_early(:,:,1) = positive_early (:,:,1)-2;
positive_early(:,:,2) = positive_early (:,:,2)+2;

positive_late  = randn(numpattern/numclass,sizebag*(1-percentage),dim);
positive_late(:,:,1)  = positive_late(:,:,1)+2;
positive_late(:,:,2)  = positive_late(:,:,2)-2;

negative = randn(numpattern/numclass,sizebag,dim)+1;


for i=1:numpattern
    patterns{1,i} = zeros(sizebag,dim);
    
    if i<=numpattern/numclass
        labels{1,i} = -ones(sizebag,1);
        patterns{1,i}(1:sizebag*percentage,1:dim) = positive_early(i,:,:);
        patterns{1,i}(sizebag*percentage+1:end,1:dim) = positive_late(i,:,:);
    else
        labels{1,i} = ones(sizebag,1);
        t = i-numpattern/numclass;
        patterns{1,i}(1:sizebag,1:dim) = negative(t,:,:);
    end
%      patterns{1,i}(1:sizebag,dim+1) = 1000;
    
end
% 
% patterns = patterns';
% labels   = labels';


if (draw==1)
    figure
    hold on
    box on
    for i=1:numpattern
        for j=1:sizebag
            if (j<=sizebag*percentage)
                if (labels{1,i}(j,1)==1)
%                 scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'MarkerEdgeColor','k','MarkerFaceColor','r')
                    earlier1 = scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'r','*');
                else
%                 scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'MarkerEdgeColor','k','MarkerFaceColor','b')
                    earlier2 = scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'b','*');
                end
            else
                if (labels{1,i}(j,1)==1)
%                 scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'MarkerEdgeColor','k','MarkerFaceColor',[1 0.6 0.6])
                    later1 = scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'m','*');
                else
%                 scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'MarkerEdgeColor','k','MarkerFaceColor',[0 0.75 0.75])
                    later2 = scatter(patterns{1,i}(j,1),patterns{1,i}(j,2),'g','*');
                end
            end
            
        end
    end
h = legend([earlier1, later1, earlier2, later2],...
    'Earliers of Class 1','Laters of Class 1','Earliers of Class 2','Laters of Class 2');
set(h,'Location','SouthEast')
title(['Number of patterns = ' num2str(numpattern) ' Size of bag = ' num2str(sizebag)])
hold off
end
% Make PATTERNS and LABELS consistent in the formation

patterns = patterns';
labels   = labels';

end