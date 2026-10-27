function S = getKSimilarity(X, Y, o, k,single)
    % GET_SIMILARITY - 计算输入数据的相似度矩阵
    % 输入：
    %   X: n * d, cell
    %   Y: n * c
    %   o: 布尔值，决定相似度公式的类型
    % 输出：
    %   A: 相似度矩阵的元胞数组
    if nargin < 5, single = 0;end
    if nargin < 2
        Y = [];  % 默认 Y 为空
    end
    if nargin < 3
        o = false;  % 默认 o 为 false
    end
    if nargin < 4 || isempty(k)
        k = [];
    end
    if size(Y,1) < size(Y,2) 
        [X, Y] = TransposeXY(X, Y);
    end

    S = cell(numel(X),1);  % 存储每个模态的相似度矩阵
    v = 1;
    % zeroflag = 0;
    while v <= numel(X)
    % for v = 1:numel(X)
        x = X{v};  % 当前数据矩阵
        n = size(x, 1);  % 数据点数量

        % 创建相同类别的掩码
        if isempty(Y)
            mask = ones(n, n);  % 无类别标签时，所有点同类别
        else
            mask = Y(:, 1) == Y(:, 1)';  % 类别相等的掩码
        end

        % 计算成对距离
        dist = sqrt(sum((reshape(x, n, 1, []) - reshape(x, 1, n, [])).^2, 3));
        
        % 计算相似度
        if o
            sim = 1 ./ (1 + dist);  % 基于距离的倒数相似度
            % zeroflag = 0;
        else
            % sigma = 1;  % 标准差
            % sim = exp(-dist ./ (2 .* sigma));  % 高斯相似度
            dist = dist.* mask;
            dist(dist == 0) = NaN;
            sigma = 0.5.*std(dist,1, 'omitnan');
            sim = exp(-dist ./ sigma);  % 高斯相似度
            sim(isnan(sim)) = 0;
        end
        sim = sim .* mask;
        sim = sim - diag(diag(sim));

        % 归一化相似度矩阵
        sim_sum = sum(sim, 1);
        sim_sum(sim_sum == 0) = 1;  % 避免除以 0
        sim = sim ./ sim_sum;

        if ~isempty(k)
            for j = 1:n  % m 是矩阵 sim 的列数
                % 获取当前列的相似度值
                col = sim(:, j);
                % 对当前列的相似度值排序，得到排序后的索引（降序排序）
                [~, ids] = sort(col, 'descend');
                % 获取前k个最相似值的索引
                top_k_indices = ids(1:k);
                % 创建一个与原列相同大小的全零列
                new_col = zeros(length(col), 1);
                % 将前k个最相似值置为原值
                new_col(top_k_indices) = col(top_k_indices);
                % 更新原始矩阵
                sim(:, j) = new_col;
            end
        end

        % 存储结果
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
        sim_sum(sim_sum == 0) = 1;  % 避免除以 0
        S = S ./ sim_sum;
    end
end
