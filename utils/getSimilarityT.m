function T = getSimilarityT(obj,X)
%-----带宽自适应且每行都归一化----------
% getSimilarityT - 计算每个模态中“特征之间”的相似度 (d×d)
% 输入：
%   X : cell，X{m} 为 n × d，n样本，d特征
% 输出：
%   T : cell，T{m} 为 d × d 的特征相似度矩阵

    M = obj.M;             % 模态数
    T = cell(M,1);

    for m = 1:M
        Xm = X{m};            % n × d
        [n, d] = size(Xm);

        % ---------- 1. 计算特征之间的 pairwise distance ----------
        % 每一列是一个 d 维特征向量（n个样本）
        % dist(i,j) = ||  Xm(:,i) - Xm(:,j)  ||
        dist = pdist2(Xm', Xm');   % d × d

        % ---------- 2. 自适应 σ ----------
        tmp = dist(:);
        tmp = tmp(tmp > 0);                   % 去掉对角线0
        sigma = 0.5 * std(tmp);               % 建议值：0.5*std

        if sigma == 0
            sigma = 1e-6;
        end

        %sigma=1;
        % ---------- 3. 高斯核相似度 ----------
        Tm = exp(- dist.^2 / (2 * sigma^2));

        % ---------- 4. 自身不相似 ----------
        Tm(1:d+1:end) = 0;

        % ---------- 5. 对称化（避免数值不对称） ----------
        Tm = (Tm + Tm') / 2;

        % ---------- 6. 归一化（每列归一化） ----------
        colsum = sum(Tm,1);
        colsum(colsum == 0) = 1;
        Tm = Tm ./ colsum;
        Tm = (Tm + Tm') / 2;

        % ---------- 7. 输出 ----------
        T{m} = Tm;
    end 
end


% function T = getSimilarityT(obj,X)
% % getSimilarityT - 计算每个模态中"特征之间"的相似度 (d×d)
% % 输入：
% %   X : cell，X{m} 为 n × d，n样本，d特征
% % 输出：
% %   T : cell，T{m} 为 d × d 的特征相似度矩阵
% 
%     M = obj.M;             % 模态数
%     T = cell(M,1);
% 
%     for m = 1:M
%         Xm = X{m};            % n × d
%         [n, d] = size(Xm);
% 
%         % ---------- 1. 计算特征之间的 pairwise distance ----------
%         dist = pdist2(Xm', Xm');   % d × d
% 
%         % ---------- 2. 固定 σ = 1 ----------
%         sigma = 1;               % 固定为1
% 
%         % ---------- 3. 高斯核相似度 ----------
%         Tm = exp(- dist.^2 / (2 * sigma^2));
% 
%         % ---------- 4. 自身不相似 ----------
%         % 如果需要对角线为0，可以取消注释
%         % Tm(1:d+1:end) = 0;
% 
%         % ---------- 5. 对称化（避免数值不对称） ----------
%         Tm = (Tm + Tm') / 2;
% 
%         % ---------- 6. 不进行归一化 ----------
%         % 注释掉以下两行
%         % colsum = sum(Tm,1);
%         % Tm = Tm ./ colsum;
% 
%         % ---------- 7. 输出 ----------
%         T{m} = Tm;
%     end 
% end

% function T = getSimilarityT(obj, X)
% % getSimilarityT - 计算每个模态中"特征之间"的相似度 (d×d)
% % 使用k-NN稀疏化
% 
%     M = obj.M;             % 模态数
%     k = 10;             % 假设从obj中获取k值
%     T = cell(M, 1);
% 
%     for m = 1:M
%         Xm = X{m};            % n × d
%         [n, d] = size(Xm);
%         
%         % 计算距离和相似度
%         dist = pdist2(Xm', Xm');   % d × d
%         sigma = 1;                  % 固定为1
%         Tm = exp(- dist.^2 / (2 * sigma^2));
%         
%         % k-NN稀疏化
%         k_actual = min(k, d-1);
%         
%         % 创建k-NN掩码
%         knn_mask = zeros(d, d);
%         
%         % 对每行找到前k_actual个最大值（排除对角线）
%         [sorted_vals, sorted_idx] = sort(Tm, 2, 'descend');
%         
%         for i = 1:d
%             % 跳过第一个（自身）
%             knn_mask(i, sorted_idx(i, 2:k_actual+1)) = 1;
%             knn_mask(i, i) = 1;  % 保持自身连接
%         end
%         
%         % 应用掩码
%         Tm = Tm .* knn_mask;
%         
%         % 对称化：如果i是j的邻居或j是i的邻居，则保留边
%         sym_mask = (Tm > 0) | (Tm' > 0);
%         Tm = Tm .* sym_mask;
%         
%         % 输出
%         T{m} = Tm;
%     end
% end
