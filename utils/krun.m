function recorder = krun(method, X, Y, params, numRepeat, numFold, varargin)
    numParams = size(params, 1);
    FeatureSpace = 1:1:10;
    % maxNF = max(featureCounts);
    seeds = randi([1, 10000], 1, numRepeat);
    verbose = 1;
    track = ParameterTracker(size(params, 2)+1, numParams);
    recorder = Recorder(numRepeat, numFold);
    repeatParams = 1;
    skipExist = 1;

    p = inputParser;
    addParameter(p, 'v', verbose);
    addParameter(p, 'nf', FeatureSpace);
    addParameter(p, 's', seeds);
    addParameter(p, 'pt', track);
    addParameter(p, 't', 30);
    addParameter(p, 'recorder', recorder);
    addParameter(p, 'rp', repeatParams);
    addParameter(p, 'cp', skipExist);
    addParameter(p, 'pr', 1);
    addParameter(p, 'task', '');
    addParameter(p, 'rateSource', 'AugLoss');
    parse(p, varargin{:});

    verbose = p.Results.v;
    FeatureSpace = p.Results.nf;
    seeds = p.Results.s;
    track = p.Results.pt;
    saveInterval = p.Results.t;
    if ~isempty(p.Results.recorder)
        recorder = p.Results.recorder;
    end  
    repeatParams = p.Results.rp;
    skipExist = p.Results.cp;
    useParfor = p.Results.pr;
    taskName = p.Results.task;
    rateSource = validatestring(p.Results.rateSource,{'Loss','AugLoss'});

    if isa(recorder.pt, 'ParameterTracker')
        track = recorder.pt;
    end
    if ~isnan(recorder.seeds)
        if  ~isnan(recorder.task)
            assert(strcmp(recorder.task, taskName), "任务不一致");
            if ~strcmp(recorder.method, method)
                warning("方法不一致，覆盖原方法");
                recorder.method = method;
                track.method = method;
            end
        else
            recorder.method = method;
        end
        seeds = recorder.seeds;
    else
        track.method = method;
        track.seeds = seeds;
        recorder.seeds = seeds;
    end
    numViews = numel(X);
    %%
    % FeatureSpace = 10;
    % FeatureSpace = ceil(10 / numViews);
    %%
    numFeatures = numel(FeatureSpace);
    maxNF = max(FeatureSpace);
    tempSpace = FeatureSpace;
    skipSpace = FeatureSpace .* numViews;

    numTotal = numRepeat*numFold;
    Wlist = cell(1,numTotal);
    Xtrain = cell(1,numTotal);
    Xtest = cell(1,numTotal);
    Ytrain = cell(1,numTotal);
    Ytest = cell(1,numTotal);
    tempResults = cell(1, numTotal);

    for i = 1:numRepeat
        [Xd, Yd] = shuffledData(X, Y, seeds(i));
        rng(seeds(i));
        cv = cvpartition(Yd(:,1), 'KFold', numFold, 'Stratify', true);
        bi = (i-1)*numFold + 1;
        ei = bi + numFold - 1;
        [Xtrain(1,bi:ei), Xtest(1,bi:ei),Ytrain(1,bi:ei), Ytest(1,bi:ei)] = cv_split(Xd, Yd, cv,true);
        % cvList{i} = cv;
    end
    seeds = repmat(seeds,1,numRepeat);
    rng('shuffle');

    skipped = 0;
    havePrintf = 0;

    for pidx = 1:numParams
        param = params(pidx,:);
        if repeatParams
            idx = find(param < 0);
            param(idx) = param(abs(param(idx)));
        end

        if skipExist && track.checkExist([skipSpace', repmat(param, numFeatures, 1)])
            if ~havePrintf
                disp('已有参数组合，将跳过。');
                havePrintf = 1;
            end
            skipped = skipped + 1;
            continue;
        end

        ins = repmat({feval(method)}, 1, numTotal);
        if isprop(ins{1},'RateSource')
            for idx = 1:numTotal
                ins{idx}.RateSource = rateSource;
            end
        end
        if isprop(ins{1},'ComputeStationarity')
            for idx = 1:numTotal
                ins{idx}.ComputeStationarity = false;
            end
        end
        tic;
        if useParfor
            parfor idx = 1:numTotal
                ins{idx} = ins{idx}.run(Xtrain{idx}, Ytrain{idx}, param, seeds(idx));
                Wlist{idx}= ins{idx}.W;
            end
        else
            for idx = 1:numTotal
                ins{idx} = ins{idx}.run(Xtrain{idx}, Ytrain{idx}, param, seeds(idx));
                Wlist{idx}= ins{idx}.W;
            end
        end
        % if ~iscell(ins{1}.W)
        %     FeatureSpace = FeatureSpace .* numViews;
        %     maxNF = max(FeatureSpace);
        % end
        if useParfor
            parfor idx = 1:numTotal
                [fxtrain, ~, fxtest] = FeatureSelection(Xtrain{idx}, Wlist{idx}, maxNF, 'X2', Xtest{idx});
                tempResults{idx} = Evaluate(fxtrain, fxtest, Ytrain{idx}, Ytest{idx}, 'kernel', 'linear','FeatureSpace',FeatureSpace);
            end
        else
            for idx = 1:numTotal
                [fxtrain, ~, fxtest] = FeatureSelection(Xtrain{idx}, Wlist{idx}, maxNF, 'X2', Xtest{idx});
                tempResults{idx} = Evaluate(fxtrain, fxtest, Ytrain{idx}, Ytest{idx}, 'kernel', 'linear','FeatureSpace',FeatureSpace);
            end
        end

        FeatureSpace = tempSpace;
        resultsList = unFlodResult(tempResults,numRepeat,numFold,numFeatures); 
        bestNf = FeatureSpace(1);
        localRe = Recorder(numRepeat, numFold);
        for i = 1:numFeatures
            nf = FeatureSpace(i);
            track = track.addParameters([nf*numViews, param], resultsList(i,:));
            tempRe = Recorder(numRepeat, numFold).log(resultsList(i,:));
            [localRe, updated] = localRe.update(tempRe);
            if updated, bestNf = nf; end
        end

        localRe.nfs = bestNf;
        localRe.ins = ins{1};
        localRe.params = param;
        localRe.RateSource = rateSource;
        for i = 1:numRepeat
            for j = 1:numFold
                fold_idx=(i-1)*numFold+j;
                fold_model=ins{fold_idx};
                [~, localRe.FSID{i,j}] = FeatureSelection(Xtrain{fold_idx}, Wlist{fold_idx}, 10);
                localRe.Loss{i,j}=fold_model.Loss;
                localRe.AugLoss{i,j}=fold_model.AugLoss;
                localRe.Err{i,j}=fold_model.Err;
                localRe.Err2{i,j}=fold_model.Err2;
                localRe.LossRate{i,j}=fold_model.LossRate;
                localRe.AugLossRate{i,j}=fold_model.AugLossRate;
                localRe.ResidualDetail{i,j}=fold_model.ResidualDetail;
                localRe.FoldConverged(i,j)=fold_model.Converged;
                localRe.Iterations(i,j)=fold_model.Iterations;
                localRe.FinalResidual(i,j)=fold_model.Err2(end);
                localRe.FinalRate(i,j)=fold_model.Err(end);
                localRe.FinalMu(i,j)=fold_model.mu;
            end
        end

        numConverged=sum(localRe.FoldConverged(:));
        recorder = recorder.update(localRe);

        [acc, stdAcc] = localRe.getM('acc');
        [globalAcc, globalStd] = recorder.getM('acc');
        elapsed = toc;

        if verbose
            fprintf(['%-3d/%-3d | %-.2fs | param: %s \t\t\t ', ...
                'acc=%-5.4f(±%-5.4f)\t bestAcc=%-5.4f(±%-5.4f)', ...
                '\t converged=%d/%d\n'], ...
                pidx, numParams, elapsed, sprintf('%-6.3g ', param), ...
                acc, stdAcc, globalAcc, globalStd,numConverged,numTotal);
        end

        if mod(pidx, saveInterval) == 0
            recorder = saveRecorderState(recorder,track,ins,seeds,method,taskName);
        end
        havePrintf = 0;
    end

    if skipped == numParams
        fprintf('所有参数均已存在于 pt 中，未执行任何计算！\n');
    else
        recorder = saveRecorderState(recorder,track,ins,seeds,method,taskName);
    end
end

function recorder = saveRecorderState(recorder,track,ins,methodSeeds,method,taskName)
    recorder.seeds = methodSeeds(1,1:10);
    recorder.method = method;
    recorder.task = taskName;
    % if isprop(ins{1,1}, 'runtime')
    %     runtimes = cellfun(@(x) x.runtime, ins(:));
    %     recorder.runtime = mean(runtimes(:));
    % end
    recorder.pt = track;
    recorder.save(['output\',taskName,'_',method,'_re_temp.mat'], 0);
end

function resultsList = unFlodResult(resultCache,numRepeat,numFold,numFeatures)
        numTotal = numel(resultCache);
        tempList.acc = NaN(numFeatures,numTotal);
        tempList.sen = NaN(numFeatures,numTotal);
        tempList.spe = NaN(numFeatures,numTotal);
        tempList.f1 = NaN(numFeatures,numTotal);
        tempList.auc = NaN(numFeatures,numTotal);
        tempList.pre = NaN(numFeatures,numTotal);
        tempList.labs = cell(numFeatures,numTotal);
        tempList.decs = cell(numFeatures,numTotal);

    for i=1:numel(resultCache)
        tempList.acc(:,i) = resultCache{i}.acc;
        tempList.sen(:,i) = resultCache{i}.sen;
        tempList.spe(:,i) = resultCache{i}.spe;
        tempList.f1(:,i) = resultCache{i}.f1;
        tempList.auc(:,i) = resultCache{i}.auc;
        tempList.pre(:,i) = resultCache{i}.pre;
        tempList.labs(:,i) = resultCache{i}.labs;
        tempList.decs(:,i) = resultCache{i}.decs;
    end

    resultsList = cell(numFeatures,numRepeat);
    for i = 1:numFeatures
        for j=1:numRepeat
            bj = (j-1)*numFold+1;
            ej = bj + numFold - 1;
            temp.acc = tempList.acc(i,bj:ej);
            temp.sen = tempList.sen(i,bj:ej);
            temp.spe = tempList.spe(i,bj:ej);
            temp.f1 =  tempList.f1(i,bj:ej);
            temp.auc = tempList.auc(i,bj:ej);
            temp.pre = tempList.pre(i,bj:ej);
            temp.labs = tempList.labs(i,bj:ej);
            temp.decs = tempList.decs(i,bj:ej);
            resultsList{i,j} = temp;
        end
    end
end
