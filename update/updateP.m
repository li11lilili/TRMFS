function P = updateP(obj,S)
%UPDATEP Auxiliary self-representation update for P=A splitting.

    P = obj.P;
    I = eye(obj.n);
    solve_options.SYM=true;
    solve_options.POSDEF=true;
    for m = 1:obj.M
        Sm = S{m};
        Em = Sm - obj.B{m} + obj.Q2{m} / obj.mu;
        numerator = Em * Sm' + obj.A{m} + obj.Q1{m} / obj.mu;
        denominator = Sm * Sm' + I;
        P{m}=linsolve(denominator,numerator',solve_options)';
    end
end
