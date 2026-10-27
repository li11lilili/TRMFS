function B = updateB_L21(obj,S)
% updateB_L21  Solve min_B 0.5*||B - A||_F^2 + lambda*||B||_{2,1}
%
% 
%
% Output:
%   B       - n×d matrix (updated according to Theorem 2)
    M=obj.M;
    B=obj.B;
    P=obj.P;
    Q2=obj.Q2;

    lambda=obj.lambda;
    mu=obj.mu;
    k=lambda/mu;

    
    
    U=cell(M,1);
    for m=1:M
        Sm=S{m};
        Pm=P{m};
        Q2m=Q2{m};
   
        U{m}=Sm-Pm*Sm+Q2m/mu;
        U1m=U{m};
        row_norm=sqrt(sum(U1m.^2,2));
        scale=max(0,1-k./max(row_norm,eps));
        B{m}=bsxfun(@times,U1m,scale);

    end
 
end
