function [w, model,iteration]=implement_decom_Learning(patterns,labels,setlossFn,type,C,subIsIn,lossMinValue)


if(~exist('setlossFn','var'))
    setlossFn = @(x,sizebag)( (1-exp(-length(x)))*sizebag/2 ); % add @20140521
end

if(~exist('isgreedy','var'))
    isgreedy = 1;
end

if(~exist('type','var'))
    type = 'lovasz';
end

% ------------------- data property  ----------------------------%
parm.patterns = patterns ;
parm.labels = labels ;
dim=size(patterns{1,1},2);

% ------------------- inference model ----------------------------%
parm.psiFn = @featureCB;
parm.sizePsi=dim;

% ------------------- loss function ------------------------------%
parm.setlossFnSub = setlossFn{1}; % f
parm.submodFnIsIncreasing = subIsIn;
parm.setlossFnSup = setlossFn{2}; % g, can be empty
parm.LossMinValue = lossMinValue;
% ------------------- constraints --------------------------------%
parm.C = C;
parm.formulationTypeSub = type{1}; % choose only for submodular f
parm.formulationTypeSup = type{2}; % choose only for supermodular g
parm.isgreedy = 1;
parm.findMostViolatedLovasz = @violateLovasz;

parm.findMostViolatedMargin = @violateMargin_greedy;
parm.findMostViolatedSlack = @violateSlack_greedy;

% -------------  customer method or parameters --------------------%
% numinbag = size(labels{1,1},1);
% parm.epsilon = 0.01;
% parm.setlossFn = setlossFn;


% ------------------------------------------------------------------
%                                                    Run SVM struct
% ------------------------------------------------------------------


[model,parm,state,iteration] = GeneralClassifier_decom(parm);

w = model.w;


end

% --------------------------------------------------------------------
%                                                SVM struct callbacks
% --------------------------------------------------------------------

function phi = featureCB(param, x, y) % gamma
% global dim
dim =size(param.patterns{1,1},2);
phi = zeros(1,dim);
for i=1:length(y) % 
    phi  = phi  +x(i,:).*y(i);    
end
phi = phi';
end

%--------------- Violate Constraint by Lovas Hinge --------------------%

function [gamma,deltaPsi] = violateLovasz(param, model, x, y,setfn)

w = model.w;

s = 1-(x*w).*y; % margin violations

[~,ind] = sort(s,'descend');
gammak = zeros(length(y),1);

for k=1:length(ind)
    lk = zeros(length(y),1);
    lk(ind(1:k)) = 1;
    lk_1 = zeros(length(y),1);
    lk_1(ind(1:k-1)) = 1;
%     gammak(k) = setfn(lk)-setfn(lk_1);
    gammak(ind(k)) = setfn(lk)-setfn(lk_1);
end

if(param.submodFnIsIncreasing)
    y(s<=0)=0;
end
gamma = sum(gammak.*double(y~=0));
% deltaPsi = ((gammak.*y(ind))'*x(ind,:))';
deltaPsi = ((gammak.*y)'*x)';

% testLovaszHingeExtension(x,y,w,setfn,gamma,deltaPsi,s(ind),ind)

end

%------------ Greedy Slack rescaling and Margin rescaling -----------------%

function yhat = violateSlack_greedy(param, model, x, y,setfn) 
% Greedy algorithm selection
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)

w = model.w;

constraint_max = -inf;% initialisation
yhat=y;% initialisation

for i=1:length(y)
    temp_y = yhat;
    temp_y(i) = temp_y(i)*(-1);
    PsiTilde = param.psiFn(param,x,temp_y);
    PsiY = param.psiFn(param,x,y); % bug fixed 12-02-2016
    constraint_new = setfn(double(temp_y~=y))*(1+dot(w,PsiTilde)-dot(w,PsiY));
    
    if constraint_new>=constraint_max
        constraint_max = constraint_new;
        yhat = temp_y;
    end
end

end
