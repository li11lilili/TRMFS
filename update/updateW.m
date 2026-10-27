function W = updateW(obj,S,Y)
    M=obj.M;
    yita=obj.yita;
    beta=obj.beta;
    alpha=obj.alpha;
    W=obj.W;
    Lt=obj.Lt;
    La=obj.La;
    solve_options.SYM=true;
    solve_options.POSDEF=true;

    for m=1:M
        Sm=S{m};
        % 计算Dm
        Wm=W{m};
        Ltm=Lt{m};
        Lam=La{m};
        h=1./(2*sqrt(sum(Wm.^2,2)+obj.epsilon_w));
        Dm=diag(h);                   % 构建对角矩阵
        
        before=Sm'*Sm+alpha*Sm'*Lam*Sm+beta*Ltm+yita*Dm;
        %before=1e-6 * eye(size(before));
        late=Sm'*Y;
        W{m}=linsolve(before,late,solve_options);
    end
end
