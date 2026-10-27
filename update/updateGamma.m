function gamma=updateGamma(obj)
    A=obj.A;
    Z=obj.Z;
    gamma=obj.gamma;
    for m=1:obj.M
        Am=A{m};
        gamma(m)=1/(2*sqrt(norm(Am-Z,'fro')^2+obj.epsilon_gamma));
    end
end
