function L = getLaplacianA(obj)
    A=obj.A;
    for m=1:obj.M
        Am = A{m};

        % === 1) 对称化（非常重要） ===
        Am = (Am + Am') / 2;
        Am(Am < 0) = 0;
        Am(1:size(Am,1)+1:end) = 0;
        % === 2) 度矩阵（行和） ===
        d = sum(Am, 2);
        D = diag(d);
    
        % === 3) 未归一化拉普拉斯 ===
        L{m} = D - Am;
    end
end
