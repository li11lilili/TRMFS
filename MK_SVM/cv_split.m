function [x_train, x_test, y_train, y_test] = cv_split(X, Y, cv,z_score)
    %
    %

    if ~exist("z_score",'var'), z_score = 0; end

    num_modalities = numel(X);
    num_folds = cv.NumTestSets;

    x_train = cell(num_folds, 1);
    x_test = cell(num_folds, 1);
    y_train = cell(num_folds, 1);
    y_test = cell(num_folds, 1);

    for fold = 1:num_folds
        train_index = cv.training(fold);
        test_index = cv.test(fold);
        
        x_train{fold} = cell(num_modalities, 1);
        x_test{fold} = cell(num_modalities, 1);
        
        y_train{fold} = Y(train_index,:);
        y_test{fold} = Y(test_index,:);
        
        for modality = 1:num_modalities
            xtrain = X{modality}(train_index, :);
            xtest = X{modality}(test_index, :);
            if z_score
                scaler = Scaler();
                [xtrain,scaler] = scaler.fit_transform(xtrain);
                xtest = scaler.transform(xtest);
            end

            x_train{fold}{modality} = xtrain;
            x_test{fold}{modality} = xtest;
        end
    end
end
