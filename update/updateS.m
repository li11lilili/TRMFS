function [S,residual] = updateS(obj,X,Y,HX)
%UPDATES 在统一目标式中更新各模态平滑表示S

    S=obj.S;
    if nargin < 4 || isempty(HX)
        HX=graphFilterS(obj,X);
    end
    residual=NaN;
    if obj.ComputeStationarity
        residual=0;
    end
    solve_options.SYM=true;
    solve_options.POSDEF=true;

    for m=1:obj.M
        Wm=obj.W{m};
        Pm=obj.P{m};
        Bm=obj.B{m};
        Lam=obj.La{m};
        Q2m=obj.Q2{m};
        Q3m=obj.Q3{m};
        Rm=eye(obj.n)-Pm;
        AS=2*eye(obj.n)+2*obj.alpha*Lam;
        CS=obj.mu*(Rm'*Rm+eye(obj.n));
        DS=2*Y*Wm'+obj.mu*Rm'*Bm-Rm'*Q2m ...
            +obj.mu*HX{m}-Q3m;

        [feature_basis,singular_values,~]=svd(Wm,'econ');
        eigenvalues=diag(singular_values).^2;
        active=eigenvalues > max(size(Wm))*eps(max([eigenvalues;1]));

        base_solution=linsolve(CS,DS,solve_options);
        Sm=base_solution;
        for q=find(active(:))'
            uq=feature_basis(:,q);
            component=linsolve(CS+eigenvalues(q)*AS,DS*uq, ...
                solve_options);
            Sm=Sm+(component-base_solution*uq)*uq';
        end
        S{m}=Sm;

        if obj.ComputeStationarity
            equation_residual=AS*(Sm*Wm)*Wm'+CS*Sm-DS;
            relative_residual=norm(equation_residual,'fro') ...
                /(norm(DS,'fro')+eps);
            residual=max(residual,relative_residual);
        end
    end
end
