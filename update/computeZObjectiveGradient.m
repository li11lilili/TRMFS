function [value,gradient,HX] = computeZObjectiveGradient(obj,Z,X, ...
    compute_gradient,Xcat,Ccat)
%COMPUTEZOBJECTIVEGRADIENT 计算Z子问题的函数值和梯度

    if nargin < 4
        compute_gradient=nargout > 1;
    end
    if nargin < 5 || isempty(Xcat)
        Xcat=horzcat(X{:});
    end
    if nargin < 6 || isempty(Ccat)
        C=obj.S;
        for m=1:obj.M
            C{m}=C{m}+obj.Q3{m}/obj.mu;
        end
        Ccat=horzcat(C{:});
    end

    n=obj.n;
    identity=eye(n);
    degree=sum(Z,2);
    degree_scale=1./sqrt(degree+obj.epsilon_D);
    G=bsxfun(@times,bsxfun(@times,Z,degree_scale),degree_scale');
    U=(identity+G)/2;

    if compute_gradient
        powers=cell(obj.p+1,1);
        powers{1}=identity;
        for k=1:obj.p
            powers{k+1}=powers{k}*U;
        end
        H=powers{obj.p+1};
    else
        H=U^obj.p;
    end

    HXcat=H*Xcat;
    HX=HXcat;
    Ecat=HXcat-Ccat;
    E=mat2cell(Ecat,n,obj.d);
    value=0;
    if compute_gradient
        MH=obj.mu*Ecat*Xcat';
    end

    for m=1:obj.M
        Em=E{m};
        value=value+obj.gamma(m)*norm(obj.A{m}-Z,'fro')^2 ...
            +obj.mu/2*norm(Em,'fro')^2;
    end

    if compute_gradient
        MH=(MH+MH')/2;
        K=zeros(n,n);
        for r=0:obj.p-1
            K=K+0.5*powers{r+1}*MH*powers{obj.p-r};
        end

        RDK=bsxfun(@times,K,degree_scale);
        diagonal_ZRDK=sum(Z.*RDK.',2);
        v=(degree+obj.epsilon_D).^(-3/2).*diagonal_ZRDK;
        direct=bsxfun(@times,bsxfun(@times,K,degree_scale),degree_scale');
        filter_gradient=direct-v*ones(1,n);
        filter_gradient=(filter_gradient+filter_gradient')/2;

        consensus_gradient=zeros(n,n);
        for m=1:obj.M
            consensus_gradient=consensus_gradient ...
                +2*obj.gamma(m)*(Z-obj.A{m});
        end
        gradient=consensus_gradient+filter_gradient;
        gradient=(gradient+gradient')/2;
        gradient(1:n+1:end)=0;
    else
        gradient=[];
    end
end
