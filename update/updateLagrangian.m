function [Q1,Q2,Q3,Q]=updateLagrangian(obj,HX,PS)
    A=obj.A_tensor;
    J=obj.J_tensor;
    Q=obj.Q_tensor;
    Q1=obj.Q1;
    Q2=obj.Q2;
    Q3=obj.Q3;
    P=obj.P;
    B=obj.B;
    S=obj.S;

    mu=obj.mu;
    for m=1:obj.M
        Am=A(:,:,m);
        Q(:,:,m)=Q(:,:,m)+mu*(Am-J(:,:,m)); 
        Q1{m}=Q1{m}+mu*(obj.A{m}-P{m});
        if nargin >= 3 && ~isempty(PS)
            PSm=PS{m};
        else
            PSm=P{m}*S{m};
        end
        Q2{m}=Q2{m}+mu*(S{m}-PSm-B{m});
        Q3{m}=Q3{m}+mu*(S{m}-HX{m});
    end
end
