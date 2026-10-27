function X=graphFilterS(obj,X)
    La=computeLaplacianLa(obj);
    I=eye(obj.n);
    H=(I-La/2)^obj.p;
    for m=1:obj.M
        X{m}=H*X{m};
    end
end
