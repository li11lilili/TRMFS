function [n, c, M, d] = getDataInfo(X, Y) % 提取样本数n、类别数c、模态数M、特征维度d
    M = numel(X);
    [n, c] = size(Y);
    d = zeros(1, M); % 预分配d的大小,1行m列
    for m = 1:M
        d(m) = size(X{m}, 2);% 遍历每个模态的样本矩阵X{m}，获取其列数，即第m个模态的特征维度
    end
end