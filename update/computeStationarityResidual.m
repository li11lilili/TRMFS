function [residual,residuals] = computeStationarityResidual( ...
    obj,Y,z_residual,s_equation_residual)

    rW=0;
    rP=0;
    rA=0;
    rB=0;
    rJ=0;
    rS=s_equation_residual;
    rGamma=0;

    for m=1:obj.M
        Sm=obj.S{m};
        Wm=obj.W{m};
        Pm=obj.P{m};
        Am=obj.A{m};
        Bm=obj.B{m};
        Lam=obj.La{m};
        Ltm=obj.Lt{m};
        Q1m=obj.Q1{m};
        Q2m=obj.Q2{m};
        Q3m=obj.Q3{m};
        Qm=obj.Q_tensor(:,:,m);

        smoothW=2*Sm'*(Sm*Wm-Y) ...
            +2*obj.alpha*Sm'*Lam*Sm*Wm+2*obj.beta*Ltm*Wm;
        kktW=l21Residual(smoothW,Wm,obj.yita);
        scaleW=1+norm(smoothW,'fro')+obj.yita*sqrt(size(Wm,1));
        rW=max(rW,norm(kktW,'fro')/scaleW);

        gradP=-Q1m-Q2m*Sm';
        scaleP=1+norm(Q1m,'fro')+norm(Q2m*Sm','fro');
        rP=max(rP,norm(gradP,'fro')/scaleP);

        Fm=Sm*Wm;
        fm2=sum(Fm.^2,2);
        Gm=0.5*(fm2+fm2'-2*(Fm*Fm'));
        Gm=max(real((Gm+Gm')/2),0);
        Gm(1:obj.n+1:end)=0;
        gradA=obj.alpha*Gm+2*obj.gamma(m)*(Am-obj.Z)+Q1m+Qm;
        stepA=1/max(1,2*obj.gamma(m));
        projectedA=projectGraph(Am-stepA*gradA);
        mapA=(Am-projectedA)/stepA;
        rA=max(rA,norm(mapA,'fro')/(1+norm(gradA,'fro')));

        gradB=-Q2m;
        kktB=l21Residual(gradB,Bm,obj.lambda);
        scaleB=1+norm(gradB,'fro')+obj.lambda*sqrt(size(Bm,1));
        rB=max(rB,norm(kktB,'fro')/scaleB);

        Rm=eye(obj.n)-Pm;
        dataS=2*(Sm*Wm-Y)*Wm';
        manifoldS=2*obj.alpha*Lam*Sm*(Wm*Wm');
        selfRepresentationS=Rm'*Q2m;
        filterS=Q3m;
        gradS=dataS+manifoldS+selfRepresentationS+filterS;
        scaleS=1+norm(dataS,'fro')+norm(manifoldS,'fro') ...
            +norm(selfRepresentationS,'fro')+norm(filterS,'fro');
        rS=max(rS,norm(gradS,'fro')/scaleS);

        gamma_gradient=norm(Am-obj.Z,'fro')^2+obj.epsilon_gamma ...
            -1/(4*obj.gamma(m)^2);
        gamma_scale=1+norm(Am-obj.Z,'fro')^2+obj.epsilon_gamma ...
            +1/(4*obj.gamma(m)^2);
        rGamma=max(rGamma,abs(gamma_gradient)/gamma_scale);
    end

    residuals=[rW,rP,rA,rB,rJ,rS,z_residual,rGamma];
    residual=max(residuals);
end


function R = l21Residual(gradient,X,weight)
    R=zeros(size(X),'like',X);
    for i=1:size(X,1)
        row_norm=norm(X(i,:),2);
        if row_norm > sqrt(eps)
            R(i,:)=gradient(i,:)+weight*X(i,:)/row_norm;
        else
            gradient_norm=norm(gradient(i,:),2);
            if gradient_norm > weight
                R(i,:)=(1-weight/gradient_norm)*gradient(i,:);
            end
        end
    end
end


function Z = projectGraph(Z)
    Z=max(real((Z+Z')/2),0);
    Z(1:size(Z,1)+1:end)=0;
end
