function L = getLaplacianA(obj)
    A=obj.A;
    for m=1:obj.M
        Am = A{m};

        Am = (Am + Am') / 2;
        Am(Am < 0) = 0;
        Am(1:size(Am,1)+1:end) = 0;
        d = sum(Am, 2);
        D = diag(d);
    
        L{m} = D - Am;
    end
end
