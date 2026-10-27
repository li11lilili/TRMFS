function Lt = getLaplacianT(obj)
%GETLAPLACIANT Feature-graph Laplacians used by the beta regularizer.

    Lt = cell(obj.M, 1);
    for m = 1:obj.M
        Tm = max(real((obj.T{m} + obj.T{m}') / 2), 0);
        Tm(1:size(Tm, 1)+1:end) = 0;
        Lt{m} = diag(sum(Tm, 2)) - Tm;
    end
end
