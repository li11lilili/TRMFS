function [Z,stationarity,HX,step] = updateZ(obj,X)
%UPDATEZ 使用投影梯度和回溯线搜索更新共识滤波图Z

    Z=projectConsensusZ(obj.Z);
    Xcat=horzcat(X{:});
    C=obj.S;
    for m=1:obj.M
        C{m}=C{m}+obj.Q3{m}/obj.mu;
    end
    Ccat=horzcat(C{:});
    [value,gradient,HXcat]= ...
        computeZObjectiveGradient(obj,Z,X,true,Xcat,Ccat);

    if isempty(obj.ZStep) || ~isfinite(obj.ZStep) || obj.ZStep <= 0
        step=1/max(1,2*sum(obj.gamma));
    else
        step=obj.ZStep;
    end
    armijo=1e-4;
    shrink=0.5;
    max_backtracking=20;

    for i=1:max_backtracking
        candidate=projectConsensusZ(Z-step*gradient);
        direction=candidate-Z;
        if norm(direction,'fro') <= eps
            Z=candidate;
            break;
        end

        [candidate_value,~,candidate_HXcat]= ...
            computeZObjectiveGradient(obj,candidate,X,false,Xcat,Ccat);
        if candidate_value <= value+armijo*sum(gradient(:).*direction(:))
            Z=candidate;
            HXcat=candidate_HXcat;
            break;
        end
        step=step*shrink;
    end

    if obj.ComputeStationarity
        [~,gradient,HXcat]= ...
            computeZObjectiveGradient(obj,Z,X,true,Xcat,Ccat);
        stationarity_step=1/max(1,2*sum(obj.gamma));
        projected=projectConsensusZ(Z-stationarity_step*gradient);
        gradient_mapping=(Z-projected)/stationarity_step;
        stationarity=norm(gradient_mapping,'fro') ...
            /(1+norm(gradient,'fro'));
    else
        stationarity=NaN;
    end
    HX=mat2cell(HXcat,obj.n,obj.d);
end


function Z = projectConsensusZ(Z)
    Z=max(real((Z+Z')/2),0);
    Z(1:size(Z,1)+1:end)=0;
end
