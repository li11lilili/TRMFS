function results = Evaluate(X_train,X_test,Y_train,Y_test, varargin)
    % profile on
    % Set default values
    % seeds = 0;
    kernel = 'linear'; 
    % kernel = {'rbf'}; 
    FeatureSpace = 1:1:10;
    % opt_weight = [0.5,0.5];
    opt_weight = ones(1, numel(X_train)) ./ numel(X_train);
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch varargin{i}
            case 'kernel'
                kernel = varargin{i+1};
            case 'FeatureSpace'
                FeatureSpace = varargin{i+1};
            otherwise
                error('Unknown parameter: %s', varargin{i});
        end
    end
    Y_train = standardizeLabels(Y_train);
    Y_test = standardizeLabels(Y_test);
    numFeatures = numel(FeatureSpace);
    results.acc = NaN(numFeatures,1);
    results.sen = NaN(numFeatures,1);
    results.spe = NaN(numFeatures,1);
    results.f1 = NaN(numFeatures,1);
    results.auc = NaN(numFeatures,1);
    results.pre = NaN(numFeatures,1);  
    results.labs = cell(numFeatures,1);
    results.decs = cell(numFeatures,1);

    for i = 1:numFeatures
        nf = FeatureSpace(i);
        [nfxtrain,nfxtest] = selectX(X_train,X_test,nf);

        k_train = mk_kernel(nfxtrain,'weights',opt_weight,'kernels',{kernel});
        model = mk_svm(k_train, Y_train);
        k_test = mk_kernel(nfxtest,'X2',nfxtrain,'weights',opt_weight,'kernels',{kernel});
        [preds, ~, decs]= mk_svm_predict(model,k_test,Y_test); % col v

        [acc, sen, spe, f1, auc,pre] = calculate_metrics(Y_test, preds, decs);
        results.acc(i,1) = acc;
        results.sen(i,1) = sen;
        results.spe(i,1) = spe;
        results.f1(i,1) = f1;
        results.auc(i,1) = auc;
        results.pre(i,1) = pre;
        results.labs{i,1} = Y_test;
        results.decs{i,1} = decs;
    end

end

function [X1,X2] = selectX(X1,X2,nf)
    for m=1:numel(X1)
        X1{m} = X1{m}(:,1:nf);
        X2{m} = X2{m}(:,1:nf);
    end
end


function Y_out = standardizeLabels(Y_in)
    if ~all(ismember(Y_in, [-1, 1]))
        Y_out = Y_in(:,1) * 2 - 1;
    else
        Y_out = Y_in(:,1);
    end
end

function [acc, sen, spe, f1, auc,pre] = calculate_metrics(y_true, y_pred, decision_values)
    cm = confusionmat(y_true, y_pred);
    tp = cm(2, 2);
    tn = cm(1, 1);
    fp = cm(1, 2);
    fn = cm(2, 1);
    tol = 1e-8;


    acc = (tp + tn) / (tp + tn + fp + fn + tol);
    sen = tp / (tp + fn + tol);
    spe = tn / (tn + fp + tol);
    pre = tp / (tp + fp + tol);
    f1 = 2 * (pre * sen) / (pre + sen + tol);
    [~, ~, ~, auc] = perfcurve(y_true, decision_values, 1);
end
