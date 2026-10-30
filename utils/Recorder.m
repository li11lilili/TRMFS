classdef Recorder
    properties
        ACC
        SEN
        SPE
        AUC
        F1
        PRE
        
        Loss = {}
        AugLoss = {}
        Err = {}
        Err2 = {}
        LossRate = {}
        AugLossRate = {}
        ResidualDetail = {}
        RateSource = 'AugLoss'
        FoldConverged
        Iterations
        FinalResidual
        FinalRate
        FinalMu
        FSID = {}
        nfs = -1;
        
        labs
        decs
        
        params
        
        r
        k

        % seeds
        evaSeeds
        seeds

        runtime = 0

        cr

        pt

        task = NaN;

        method = '';

        ins
    end
    
    methods
        function obj = Recorder(r, k, params)
            if nargin > 0
                obj.cr = 0;
                obj.r = r;
                obj.k = k;
                
                obj.ACC = NaN(r, k);
                obj.SEN = NaN(r, k);
                obj.SPE = NaN(r, k);
                obj.AUC = NaN(r, k);
                obj.F1 = NaN(r, k);
                obj.PRE = NaN(r, k);

                obj.evaSeeds = [];
                % obj.seeds = NaN(1,r);
                obj.seeds = [];

                obj.Loss = cell(r,k);
                obj.AugLoss = cell(r,k);
                obj.Err = cell(r,k);
                obj.Err2 = cell(r,k);
                obj.LossRate = cell(r,k);
                obj.AugLossRate = cell(r,k);
                obj.ResidualDetail = cell(r,k);
                obj.FoldConverged = false(r,k);
                obj.Iterations = NaN(r,k);
                obj.FinalResidual = NaN(r,k);
                obj.FinalRate = NaN(r,k);
                obj.FinalMu = NaN(r,k);
                obj.FSID = cell(r,k);
                
                obj.labs = cell(r,k);
                obj.decs = cell(r,k);
                
                if nargin > 2
                    obj.params = params;
                else
                    obj.params = [];
                end
            end
        end

        function obj = log(obj, results)
            if iscell(results)
                for i=1:numel(results)
                    result = results{i};
                    obj.cr = obj.cr + 1;
                    cr_ = obj.cr;
%                     disp('---- Logging result ----');
%                     disp(result);
                    obj.ACC(cr_, :) = result.acc;
                    obj.SEN(cr_, :) = result.sen;
                    obj.SPE(cr_, :) = result.spe;
                    obj.AUC(cr_, :) = result.auc;
                    obj.F1(cr_, :) = result.f1;
                    obj.PRE(cr_, :) = result.pre;
                    for ck = 1:obj.k
                        obj.labs{cr_,ck} = result.labs{ck};        
                        obj.decs{cr_,ck} = result.decs{ck}; 
                    end
                end
            else
                obj.cr = obj.cr + 1;
                cr_ = obj.cr;
                obj.ACC(cr_, :) = results.acc;
                obj.SEN(cr_, :) = results.sen;
                obj.SPE(cr_, :) = results.spe;
                obj.AUC(cr_, :) = results.auc;
                obj.F1(cr_, :) = results.f1;
                obj.PRE(cr_, :) = results.pre;
                obj.evaSeeds = results.seeds; 
                for ck = 1:obj.k
                    obj.labs{cr_,ck} = results.labs{ck};        
                    obj.decs{cr_,ck} = results.decs{ck}; 
                end
            end
        end

        function [value, stdV] = getMetrics(obj,type)
            switch lower(type)
                case 'acc'
                    data = obj.ACC;
                case 'sen'
                    data = obj.SEN;
                case 'spe'
                    data = obj.SPE;
                case 'auc'
                    data = obj.AUC;
                case 'f1'
                    data = obj.F1;
                case 'pre'
                    data = obj.PRE;    
            end
            if any(isnan(data))
                waring('there is nan in data, use 0 to replace!\n')
            end
            data(isnan(data)) = 0;
            value = mean(data,2);
            stdV = std(data,[],2);
        end
        

        
        function save(obj, filename,~)
            [parentDir, ~, ~] = fileparts(filename);
            if ~isfolder(parentDir) % ~exist(parentDir, 'dir')
                mkdir(fullfile(pwd,parentDir));
            end
            save(filename, 'obj');
            if nargin < 3
                fprintf('已将模型保存到文件: %s\n', filename);
            end
        end
        
        function obj = load(~,filename)
            data = load(filename);
            obj = data.obj;
            fprintf('已从 MAT 文件加载数据: %s\n', filename);
        end

        function [value, stdV] = getM(obj, type)
            switch lower(type)
                case 'acc'
                    cleaned_data = obj.ACC(~isnan(obj.ACC));
                case 'sen'
                    cleaned_data = obj.SEN(~isnan(obj.SEN));
                case 'spe'
                    cleaned_data = obj.SPE(~isnan(obj.SPE));
                case 'auc'
                    cleaned_data = obj.AUC(~isnan(obj.AUC));
                case 'f1'
                    cleaned_data = obj.F1(~isnan(obj.F1));
                case 'pre'
                    cleaned_data = obj.PRE(~isnan(obj.PRE));
                otherwise
                    error('Invalid type. Valid types are: ACC, SEN, SPE, AUC, F1.');
            end
            if isempty(cleaned_data)
                value = 0;
                stdV = 0;
            else
                value = mean(cleaned_data);
                stdV = std(cleaned_data);
            end

        end

        function printMetrics(obj)
            metrics = {'ACC', 'SEN', 'SPE', 'F1', 'AUC','PRE'};
            
            fprintf('-----------------------------------------\n');
            fprintf('| Item \tMean\tStd \t|\n');
            fprintf('-----------------------------------------\n');
            
            for i = 1:length(metrics)
                type = metrics{i};
                [value, stdV] = obj.getM(type);
                fprintf('| %-4s \t%5.4f\t%5.4f\t|\n', type, value, stdV);
            end
            
            fprintf('-----------------------------------------\n');
        end

        function [obj,flag] = update(obj,o1)
            flag = 0;
            if isempty(obj.ACC) || ~any(~isnan(obj.ACC(:)))
                obj=o1;
                flag=1;
                return;
            end
            [v, ~] = obj.getM('acc');
            [v1,~] = o1.getM('acc');
            if v < v1 
                obj = o1;
                flag = 1; 
            elseif abs(v - v1) < 1e-5
                [va, ~] = obj.getM('auc');
                [v1a,~] = o1.getM('auc');
                [vf, ~] = obj.getM('f1');
                [v1f,~] = o1.getM('f1');
                if va < v1a 
                    obj = o1;
                    flag = 1;
                elseif abs(va - v1a)< 1e-5 && vf < v1f
                    obj = o1;
                    flag = 1;
                elseif o1.nfs < obj.nfs && abs(va - v1a)< 1e-5 && abs(vf - v1f )< 1e-5
                    obj = o1;
                    flag = 1;
                end
            end
        end

        function [labslist,decslist] = plotROC(obj, r, k)
            if nargin < 3, k = obj.k;end
            if nargin < 2, r = obj.r;end
            labslist = [];
            decslist = [];
            for ridx  = 1:r
                for kIdx = 1:k
                    if ~isempty(obj.labs{r, kIdx}) && ~isempty(obj.decs{r, kIdx})
                        labslist = [labslist; obj.labs{r, kIdx}(:)];
                        decslist = [decslist; obj.decs{r, kIdx}(:)];
                    end
                end
            end

            if isempty(decslist) || isempty(labslist)
                fprintf('Error: Either decs or preds is empty.\n');
                return
            end
            [fpr, tpr, ~, ~] = perfcurve(labslist(:), decslist(:), 1);
            
            figure;
            % plot(fpr, tpr, 'o-', 'LineWidth', 1);
            %             colors = ColorMap(2);
            colors = parula(2);
            plot(fpr, tpr, 'Color',colors(1,:), 'LineWidth', 1);
            title(['ROC Curve for Experiment ' num2str(r)]);
            xlabel('False Positive Rate (FPR)');
            ylabel('True Positive Rate (TPR)');
            % grid off;
        end

        function loss = plotLoss(obj,be,r,k,color)
            if nargin < 2, be = 1; end
            if nargin < 3, r = 1; end
            if nargin < 4, k = 0; end
            if nargin < 5, color = parula(10); end

            if k == 0
                loss = obj.Loss(r,:);
            else
               loss = obj.Loss(r,k);
            end

            figure;
            hold on;
            for i = 1:numel(loss)
                L = loss{i};      
                L = L(:,be:end);  
                len = size(L, 2);
                if iscell(color) 
                    c = color{i};
                elseif r ~= 0 && isscalar(color)
                    c = color;
                else
                    c = color(i,:);
                end
                plot(1:len, L, 'Color', c, ...
                    'Marker', 'o','MarkerSize',3 ,'LineWidth', 1, 'DisplayName', ['Loss ' num2str(i)]);
            end
            hold off; 
            legend show; 
            xlabel('Epoch'); 
            ylabel('Original Objective Value');
            title('Original Objective Value');
        end

        function loss = plotAugLoss(obj,be,r,k,color)
            if nargin < 2, be = 1; end
            if nargin < 3, r = 1; end
            if nargin < 4, k = 0; end
            if nargin < 5, color = parula(10); end

            if k == 0
                loss = obj.AugLoss(r,:);
            else
                loss = obj.AugLoss(r,k);
            end

            figure;
            hold on;
            for i = 1:numel(loss)
                L = loss{i};
                if isempty(L), continue; end
                L = L(:,be:end);
                len = size(L,2);
                if iscell(color)
                    c = color{i};
                elseif r ~= 0 && isscalar(color)
                    c = color;
                else
                    c = color(i,:);
                end
                plot(1:len,L,'Color',c,'Marker','o', ...
                    'MarkerSize',3,'LineWidth',1, ...
                    'DisplayName',['AugLoss ' num2str(i)]);
            end
            hold off;
            legend show;
            xlabel('Epoch');
            ylabel('Augmented Lagrangian Value');
            title('Augmented Lagrangian (Diagnostic Only)');
        end

        function plotConvergence(obj,r,k)
            if nargin < 2, r = 1; end
            if nargin < 3, k = 1; end
            if isempty(obj.Err) || isempty(obj.Err2) || ...
                    size(obj.Err,1) < r || size(obj.Err,2) < k || ...
                    size(obj.Err2,1) < r || size(obj.Err2,2) < k
                warning('Recorder:NoConvergenceHistory', ...
                    'No convergence history is available for this fold.');
                return;
            end
            residual=obj.Err2{r,k};
            rate=obj.Err{r,k};
            if isempty(residual) || isempty(rate)
                warning('Recorder:NoConvergenceHistory', ...
                    'No convergence history is available for this fold.');
                return;
            end
            figure;
            semilogy(1:numel(residual),residual,'LineWidth',1.5);
            hold on;
            semilogy(1:numel(rate),rate,'LineWidth',1.5);
            yline(1e-3,'--','Tolerance');
            hold off;
            grid on;
            xlabel('Iteration');
            ylabel('Residual / Relative Change');
            legend('Infinity-norm primal residual',...
                [obj.RateSource ' relative change'], ...
                'Location','best');
            title(sprintf('Convergence Diagnostics (repeat %d, fold %d)',r,k));
        end
    end
   
end
