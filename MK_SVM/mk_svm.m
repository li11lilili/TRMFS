function model = mk_svm(X, Y,options)
    
    if nargin < 3
        options = '-t 4 -q';
    end
    
    N = size(Y,1);
    X_ = [(1:N)', X];

    model = svmtrain(Y, X_, options);
end
