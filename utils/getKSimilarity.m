function S = getKSimilarity(X, Y, o, k,single)
    %   X: n * d, cell
    %   Y: n * c
    if nargin < 5, single = 0;end
    if nargin < 2
        Y = [];
    end
    if nargin < 3
        o = false;
    end
    if nargin < 4 || isempty(k)
        k = [];
    end
    if size(Y,1) < size(Y,2) 
        [X, Y] = TransposeXY(X, Y);
    end

    S = cell(numel(X),1);
    v = 1;
    % zeroflag = 0;
    while v <= numel(X)
    % for v = 1:numel(X)
        x = X{v};
        n = size(x, 1);

        if isempty(Y)
            mask = ones(n, n);
        else
            mask = Y(:, 1) == Y(:, 1)';
        end

        dist = sqrt(sum((reshape(x, n, 1, []) - reshape(x, 1, n, [])).^2, 3));
        
        if o
            sim = 1 ./ (1 + dist);
            % zeroflag = 0;
        else
            dist = dist.* mask;
            dist(dist == 0) = NaN;
            sigma = 0.5.*std(dist,1, 'omitnan');
            sim = exp(-dist ./ sigma);
            sim(isnan(sim)) = 0;
        end
        sim = sim .* mask;
        sim = sim - diag(diag(sim));

        sim_sum = sum(sim, 1);
        sim_sum(sim_sum == 0) = 1;
        sim = sim ./ sim_sum;

        if ~isempty(k)
            for j = 1:n
                col = sim(:, j);
                [~, ids] = sort(col, 'descend');
                top_k_indices = ids(1:k);
                new_col = zeros(length(col), 1);
                new_col(top_k_indices) = col(top_k_indices);
                sim(:, j) = new_col;
            end
        end

        S{v} = sim;
        v = v + 1;
    end


    if single
        G = 0;
        for m=1:numel(X)
            G = G + S{m};
        end
        G = G ./ numel(X);
        S = G;
        sim_sum = sum(S, 1);
        sim_sum(sim_sum == 0) = 1;
        S = S ./ sim_sum;
    end
end
