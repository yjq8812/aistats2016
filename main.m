function [err] = main
% An example function for a surrogate operator for non-odular loss functions
% --
% Implemented by 
% Jiaqian Yu & Matthew B.Blashcko @ 2016
% 
% For more details: 
% Yu, J. and M. B. Blaschko: A Convex Surrogate Operator for General Non-Modular 
% Loss Functions. AISTATS , 2016

% load/generate data
[X,Y] = genData();

% an example loss function
% given its decomposition i.e. l = f + g
[f,g] = testLossFunc(1);

h = @(x)(length(find(x)));% hamming loss
ourhamming ={h;[]};
ourloss =  {f;g};
subIsIn = 0; % whether the submodular is increasing
lossMinValue = 0; % give the minimum value of the loss by hand


% Cross Validation --------------------------%
n = length(X);

Xtrain   = X(1:ceil(n/2));
Ytrain   = Y(1:ceil(n/2));
Xtest   = X(ceil(n/2)+1:end);
Ytest   = Y(ceil(n/2)+1:end);

C = 1;% regularizer
      % cross-validation for C values to be implented by user
  
% ---------------------- RE-TRAINING & TESTING-----------------------------%
% err = testEval(Xtest,Ytest,w,ourloss{1})+ testEval(Xtest,Ytest,w,ourloss{2});

fprintf(['** Training... **\n']);
[w_decom,~,iteration_decom] = trainDecomposedLoss(Xtrain,Ytrain,ourloss,{'lovasz';'slack'},C,subIsIn,lossMinValue);
[w_slack,~,iteration_slack] = trainSlackRescaling(Xtrain,Ytrain,ourloss,C,subIsIn,lossMinValue);
    
% test on test
% [Xtest,Ytest] = genData(cls,'test');
% [Xtest,Ytest] = generateCOCO(50,'val');
fprintf(['**  Testing...    \n']);
err = zeros(2,2);
err(1,1) = testEval(Xtest,Ytest,w_decom, ourloss{1})   + testEval(Xtest,Ytest,w_decom, ourloss{2});
err(2,1) = testEval(Xtest,Ytest,w_slack, ourloss{1})   + testEval(Xtest,Ytest,w_slack, ourloss{2});

err(1,2) = testEval(Xtest,Ytest,w_decom,h);
err(2,2) = testEval(Xtest,Ytest,w_slack, h);
end


function [err,errList] = testEval(X,Y,w,lossfn)
if isempty(lossfn)
    err = 0;
    errList = 0;
else
    for i=1:length(X)
        errList(i) = lossfn(double(sign(X{i}*w)~=Y{i}));
    end
    
    err = mean(errList);
end
end


function [X,Y] = genData()
% Generate/Load your data here
% X : patterns, a cell in size of n*1; each cell in size of p*d
% Y : labels, a cell in size of n*1; each cell in size of p*1
% n : number of patterns
% p : size of bags
% d : dimension of feature vectors

% use a sythetic data for example
[X,Y] = generateSyntheticData(1000,10);
end

function [w,model,iteration] = trainDecomposedLoss(X,Y,lossfn,type,C,subIsIn,lossMinValue)

[w,model,iteration]=...
    implement_decom_Learning(X,Y,lossfn,type,C,subIsIn,lossMinValue);
end

function [w,model,iteration] = trainSlackRescaling(X,Y,lossfn,C,subIsIn,lossMinValue)
[w,model,iteration]=...
    implement_decom_Learning(X,Y,lossfn,{'slack';[]},C,subIsIn,lossMinValue);
end


