function [residual,residuals] = computePrimalResidual(obj,X,HX,PS)
%COMPUTEPRIMALRESIDUAL 计算四类约束的矩阵无穷范数残差
% MATLAB 对矩阵的 norm(R,inf) 返回最大绝对行和，
% 与 RMoGL 公开源码的约束残差计算方式一致。

    if nargin < 3 || isempty(HX)
        HX=graphFilterS(obj,X);
    end

    r1=0;
    r2=0;
    r3=0;
    r4=0;
    for m=1:obj.M
        Am=obj.A{m};
        Pm=obj.P{m};
        Sm=obj.S{m};
        Bm=obj.B{m};
        Jm=obj.J_tensor(:,:,m);

        term1=Am-Pm;
        if nargin >= 4 && ~isempty(PS)
            PSm=PS{m};
        else
            PSm=Pm*Sm;
        end
        term2=Sm-PSm-Bm;
        term3=obj.A_tensor(:,:,m)-Jm;
        term4=Sm-HX{m};

        r1=max(r1,norm(term1,inf));
        r2=max(r2,norm(term2,inf));
        r3=max(r3,norm(term3,inf));
        r4=max(r4,norm(term4,inf));
    end
    residuals=[r1,r2,r3,r4];
    residual=max(residuals);
end
