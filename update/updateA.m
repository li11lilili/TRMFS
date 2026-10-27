function A=updateA(obj,S)

    gamma=obj.gamma;
    mu=obj.mu;

    M=obj.M;
    
    A=obj.A;
    Z=obj.Z; 
    P=obj.P;
    J=obj.J_tensor;
    Q=obj.Q_tensor;
    Q1=obj.Q1;

    for m=1:M
        Sm=S{m};
        Wm=obj.W{m};
        Pm=P{m};
        Jm=J(:,:,m);
        Qm=Q(:,:,m);
        Q1m=Q1{m};

        Fm=Sm*Wm;
        fm2=sum(Fm.^2,2);
        Gm=0.5*(fm2+fm2'-2*(Fm*Fm'));
        Gm=max(real((Gm+Gm')/2),0);
        Gm(1:obj.n+1:end)=0;

        late=2*gamma(m)*Z+mu*Pm+mu*Jm-Q1m-Qm-obj.alpha*Gm;
        Am=late/(2*gamma(m)+2*mu);
        Am=(Am+Am')/2;
        Am=max(real(Am),0);
        Am(1:obj.n+1:end)=0;
        A{m}=Am;             
    end
end
