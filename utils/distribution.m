function distribution(Y)
    if istable(Y)
        data = Y;
        [groupUnique, ~, groupIndices] = unique(data.Group);

        groupCounts = histcounts(groupIndices, 1:length(groupUnique)+1);

        disp('Group counts:');
        for i = 1:length(groupUnique)
            fprintf('%s: %d\n', groupUnique{i}, groupCounts(i));
        end
        return
    end
    
    disp('-------------------------------------');
    counts = tabulate(Y(:,1));
    counts = sortrows(counts, 1,'descend');
    disp('Class Distribution:');
    for j = 1:size(counts, 1)
        fprintf('%g\t%g\t%.2f%%\n', counts(j, 1), counts(j, 2), counts(j, 3));
    end
    disp('-------------------------------------');
end

