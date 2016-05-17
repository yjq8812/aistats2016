function [model,sparm,state,iteration] = GeneralClassifier_decom(sparm, oldstate)
% modification: 15-May-2015, function for NIPS2015 decomposer loss function
%
state = bundler(); % initialize state
state.lambda = 1 ./ (sparm.C);


if (~isfield(sparm,'convergenceThreshold'))
    sparm.convergenceThreshold = 0.008;
end

maxIterations = 1000;

sparm.w = zeros(sparm.sizePsi,1);
state.w = sparm.w;

model.w = state.w;

if (exist('oldstate','var'))
    for i=1:length(oldstate.b)
        if(oldstate.softVariables(i))
            state = bundler(state,oldstate.a(:,i),oldstate.b(i));
        end
    end
end

%%%%%%%% put the constraint with the minimum value of the loss
a = zeros(sparm.sizePsi,1);
b = sparm.LossMinValue;
state = bundler(state,a,b);


minIterations = 10;
numIterations = 0;

bestPrimalObjective = Inf;

iteration.iter = 0;
iteration.gap = [];

while (((bestPrimalObjective - state.dualObjective)/state.dualObjective > sparm.convergenceThreshold ...
        || minIterations>0) && numIterations < maxIterations )
%     tic
    
    numIterations = numIterations + 1;
    minIterations = minIterations - 1;
 
    %------------------------------ Decompostion ------------------------------%
    if size(sparm.formulationTypeSub)~=0
        switch sparm.formulationTypeSub
            case 'lovasz'
                tic
                [phi_g, b_g] = computeOneslackLovasz(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSub);
                toc
            case 'margin'
                tic
                [phi_g, b_g] = computeOneslackMargin(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSub,0);
                toc
            case 'slack'
                [phi_g, b_g] = computeOneslackSlack(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSub,0);
            otherwise
                error('The type has not been well defined for the submodular loss!')
        end
    else
        phi_g = 0;
        b_g = 0;
    end
    
    if size(sparm.formulationTypeSup)~=0
        switch sparm.formulationTypeSup
            case 'lovasz'
                error('Lovasz hinge cannot work with supermodular increasing set function!')
            case 'margin'
                tic
                [phi_h, b_h] = computeOneslackMargin(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSup,1);
                toc
            case 'slack'
                [phi_h, b_h] = computeOneslackSlack(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSup,1);
            otherwise
                error('The type has not been well defined for the supermodular loss!')
        end
    else
        phi_h = 0 ;
        b_h = 0 ;
    end
    phi = phi_g +phi_h;
    b = b_g + b_h;

    %------------------------------ End of decomp. ------------------------------%
    
    if (norm(phi)==0)
        phi= zeros(size(state.w));        
        fprintf('\n No New Violated Constraint Added!!\n\n');
    end
    
    primalobjective = (state.lambda / 2) * (state.w' * state.w) + b - dot(state.w, phi);
    if (primalobjective < bestPrimalObjective)
        bestPrimalObjective = primalobjective;
        bestState = state;
    end
    
    gap = (bestPrimalObjective - state.dualObjective) / state.dualObjective;
    
    fprintf([' %d primal objective: %f, best primal: %f, dual objective: %f, gap: %f\n'], ...
        numIterations, primalobjective, bestPrimalObjective, state.dualObjective,gap);
    
    state = bundler(state, phi, b);
    sparm.w = state.w;
    model.w = state.w;
       
    iteration.iter = numIterations;
    iteration.gap = [iteration.gap gap];
    
    if norm(model.w)==0
        fprintf('\n WARNING: Learned Weight Vector is Empty!!!\n\n');
    end
%     toc
end


sparm.w = bestState.w;
model.w = bestState.w;

end



function [phi, b] = computeOneslackLovasz(sparm,model,X,Y,setfn)
phi = 0;
b = 0;

% For each pattern
for i = 1 : length(X);
    [gamma,deltaPsi] = sparm.findMostViolatedLovasz(sparm, model, X{i}, Y{i},setfn);
    if (gamma - dot(model.w,deltaPsi) > 0 )
        b = b + gamma;
        phi = phi + deltaPsi;
    end
end

end


function [phi, b] = computeOneslackSlack(sparm,model,X,Y,setfn,issuper)
phi = 0;
b = 0;

% For each pattern
for i = 1 : length(X);
    
    [tildeY] = sparm.findMostViolatedSlack(sparm, model, X{i}, Y{i},setfn);
    
    delta = setfn(Y{i}~=tildeY);%(y,ybar)
    
    deltaPsi =  sparm.psiFn(sparm, X{i}, Y{i}) - sparm.psiFn(sparm,X{i},tildeY);
    
    if (delta*(1 -  dot(model.w,deltaPsi)) > eps)
        b = b + delta;
        phi = phi + deltaPsi;
    end
    
end

% phi = phi';
end
