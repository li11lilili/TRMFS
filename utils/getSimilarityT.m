function T = getSimilarityT(obj,X)

    M = obj.M;
    T = cell(M,1);

    for m = 1:M
        Xm = X{m};            % n × d
        [n, d] = size(Xm);

        % dist(i,j) = ||  Xm(:,i) - Xm(:,j)  ||
        dist = pdist2(Xm', Xm');   % d × d

        tmp = dist(:);
        tmp = tmp(tmp > 0);
        sigma = 0.5 * std(tmp);

        if sigma == 0
            sigma = 1e-6;
        end

        %sigma=1;
        Tm = exp(- dist.^2 / (2 * sigma^2));

        Tm(1:d+1:end) = 0;

        Tm = (Tm + Tm') / 2;

        colsum = sum(Tm,1);
        colsum(colsum == 0) = 1;
        Tm = Tm ./ colsum;
        Tm = (Tm + Tm') / 2;

        T{m} = Tm;
    end 
end


% function T = getSimilarityT(obj,X)
% 
%     T = cell(M,1);
% 
%     for m = 1:M
%         Xm = X{m};            % n × d
%         [n, d] = size(Xm);
% 
%         dist = pdist2(Xm', Xm');   % d × d
% 
% 
%         Tm = exp(- dist.^2 / (2 * sigma^2));
% 
%         % Tm(1:d+1:end) = 0;
% 
%         Tm = (Tm + Tm') / 2;
% 
%         % colsum = sum(Tm,1);
%         % Tm = Tm ./ colsum;
% 
%         T{m} = Tm;
%     end 
% end

% function T = getSimilarityT(obj, X)
% 
%     T = cell(M, 1);
% 
%     for m = 1:M
%         Xm = X{m};            % n × d
%         [n, d] = size(Xm);
%         
%         dist = pdist2(Xm', Xm');   % d × d
%         Tm = exp(- dist.^2 / (2 * sigma^2));
%         
%         k_actual = min(k, d-1);
%         
%         knn_mask = zeros(d, d);
%         
%         [sorted_vals, sorted_idx] = sort(Tm, 2, 'descend');
%         
%         for i = 1:d
%             knn_mask(i, sorted_idx(i, 2:k_actual+1)) = 1;
%         end
%         
%         Tm = Tm .* knn_mask;
%         
%         sym_mask = (Tm > 0) | (Tm' > 0);
%         Tm = Tm .* sym_mask;
%         
%         T{m} = Tm;
%     end
% end
