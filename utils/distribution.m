function distribution(Y) % MRI.scv变为表格后（缺失数据已赋0）
    if istable(Y)
        data = Y;
        % 获取 Group 列中的唯一组别:AD\CN\LMCI...
        [groupUnique, ~, groupIndices] = unique(data.Group);

        % 统计每个组别的数量
        groupCounts = histcounts(groupIndices, 1:length(groupUnique)+1);

        % 打印每个组别的数量
        disp('Group counts:');
        for i = 1:length(groupUnique)
            fprintf('%s: %d\n', groupUnique{i}, groupCounts(i));
        end
        return
    end
    
    disp('-------------------------------------');
    counts = tabulate(Y(:,1));
    counts = sortrows(counts, 1,'descend');  % 第2列是频数
    disp('Class Distribution:');
    for j = 1:size(counts, 1)
        fprintf('%g\t%g\t%.2f%%\n', counts(j, 1), counts(j, 2), counts(j, 3));
    end
    disp('-------------------------------------');
end

