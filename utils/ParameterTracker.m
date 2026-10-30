classdef ParameterTracker
    properties
        params
        metrics
        seeds
        numGroups
        maxGroups
        numP
        method
    end
    
    methods
        function obj = ParameterTracker(NumP,MaxGroups)
            if nargin < 2
                MaxGroups = 1000;
            end
            obj.numP = NumP;
            obj.maxGroups = MaxGroups;
            obj.numGroups = 0;
            obj.params = zeros(MaxGroups, NumP);
            obj.metrics = zeros(MaxGroups, 6);
            obj.seeds = NaN;
        end

        function [obj,acc] = addParameters(obj, param_values,result)
            if iscell(result)
                mes = zeros(numel(result),6);
                for r = 1:numel(result)
                    re = result{r};
                    tm = [
                        mean(re.acc,'omitnan'), ... 
                        mean(re.sen,'omitnan'), ...
                        mean(re.spe,'omitnan'), ...
                        mean(re.f1,'omitnan'),  ...
                        mean(re.auc,'omitnan'),  ...
                        mean(re.pre,'omitnan')  ...
                    ];
                    mes(r,:) = tm;
                end
                m = mean(mes, 1, 'omitnan');
            else
                m = result;
            end
            acc = m(1);
            idx = findParam(obj,param_values);
            if idx ~= 0
                obj.params(idx, :) = param_values;
                obj.metrics(idx, :) = m;
            else
                if obj.numGroups >= obj.maxGroups
                    obj.maxGroups = obj.maxGroups * 2;
                    obj.params(obj.numGroups + 1:obj.maxGroups, :) = 0;
                    obj.metrics(obj.numGroups + 1:obj.maxGroups, :) = 0; 
                end
                obj.numGroups = obj.numGroups + 1;
                obj.params(obj.numGroups, :) = param_values;
                obj.metrics(obj.numGroups, :) = m;
            end
        end
        
        function obj = addMetrics(obj, metrics)
            if obj.numGroups == 0
                error('没有参数组，请先添加参数组');
            end
            obj.metrics(obj.numGroups, :) = metrics;
        end

        function rowIndex = findLike(~,ipdata,param)
            rowIndex = ismember(ipdata, param, 'rows'); 
        end

        function [maxlist,maxparam] = plotAblation(obj,colors,leng,metricIdx)
            nump = obj.numP;
            maxlist = zeros(1,nump+1);
            
            if nargin >= 2 && isempty(colors)
                colors = checkColor(colors,nump+10);
            end
            if nargin < 2
                colors = checkColor([],nump+10);
            end

            if nargin < 4, metricIdx=1; end

            data = [obj.params(1:obj.numGroups,:),obj.metrics(1:obj.numGroups,metricIdx)];
            
            figure;
            hold on;
            for ip=2:nump+1
                ipdata = data(data(:,ip) == 0,:);
                if size(ipdata,1) == 0
                    fprintf('continue, no zero to draw in idx: %d\n', ip-1);
                    continue
                end
                ipdata = sortrows(ipdata,'descend');
                [maxlist(ip-1), maxIndex] = max(ipdata(:, end));

                param = ipdata(maxIndex,2:end-1);
                rowIndex = obj.findLike(ipdata(:,2:end-1),param);
                
                ipdata = ipdata(rowIndex,[1, end]);
                ipdata = sortrows(ipdata,'ascend');
                if iscell(colors)
                    c = colors{ip-1};
                else
                    c = colors(ip-1,:);
                end
                if nargin < 3
                    lg = ['no-',num2str(ip-1)];
                else
                    lg = leng{ip-1};
                end
                plot(ipdata(:,1), ipdata(:,2), 'Color', c,'Marker', '.', ...
                    'MarkerSize',1.3 ,'LineWidth', 1.2,'DisplayName', lg);
            end
            
            [maxlist(end), maxIndex] = max(data(:, end));
            param = data(maxIndex,2:end-1);
            nfs = data(maxIndex,1);
            maxparam = [nfs,param];
            id = find(param == 0);
            if ~isempty(id)
                warning('The max value is in paramIdx = %s', mat2str(id));
            else
                if iscell(colors)
                    c = colors{nump+1};
                else
                    c = colors(nump+1,:);
                end
                if nargin >= 3 && numel(leng) >= nump+1
                    lg = leng{nump+1};
                else
                    lg = 'all';
                end
                rowIndex = obj.findLike(data(:,2:end-1),param);
                data = data(rowIndex,[1, end]);
                data = sortrows(data,'ascend');
                plot(data(:,1), data(:,2), 'Color', c,'Marker', '.', ...
                'MarkerSize',1.3 ,'LineWidth', 1.2,'DisplayName', lg);
            end
            legend('show');
            legend('Location', 'best');
            
        end
        
        function plotBar(obj, params,pName, metricIdx)
            %
            %   params = [-1, -1, 0.5, 0.6, 0.7, 0.8];
            %   metricIdx = 1;
            %   pName = {'𝛼', '𝛽', 'acc'};
            %   obj.plotParamBar(params,pName,metricIdx);
            if nargin < 4, metricIdx = 1; end
            idx = find(params == -1);
            assert(numel(idx) == 2, 'The params needing to plot must be exactly 2 (-1).');
        
            fixid = (params ~= -1);
            fixparam = params(fixid);
            rowIndex = obj.findLike(obj.params(:,fixid), fixparam);
        
            data = [obj.params(1:obj.numGroups, ~fixid), obj.metrics(1:obj.numGroups, metricIdx)];
            data = data(rowIndex, :);
        
            fp1 = data(:, 1);
            fp2 = data(:, 2);
            m = data(:, 3);

            [X, Y] = meshgrid(unique(fp1), unique(fp2));
            Z = zeros(size(X));
        
            for i = 1:size(data, 1)
                xIdx = X(1, :) == fp1(i);
                yIdx = Y(:, 1) == fp2(i);
                Z(yIdx, xIdx) = m(i);
            end

            figure;
            hBar  = bar3(Z);
            if nargin < 3
                pName = {['param',num2str(idx(1))],['param',num2str(idx(2))],'Metric'};
            end
            xlabel(pName{1});
            ylabel(pName{2});
            zlabel(pName{3});

            % title(['3D Bar Plot of Metric ', num2str(metricIdx)]);
    
            set(gca, 'XTickLabel', string(unique(fp1)), 'YTickLabel', string(unique(fp2)), ...
                'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'normal');
            zlim([0, 1]);
            set(gca, 'ZTick', 0:0.2:1),
            grid on;
            colormap(jet);
            % setBarColor(hBar,1);
        end

        function [p,m] = plotEffect(obj, params, metricIdx)
            if nargin < 3, metricIdx=1; end
            if metricIdx < 1 || metricIdx > 5
                error('Invalid metric index');
            end
            
            plotIdx = find(params == -1);
            if numel(plotIdx) ~= 1
                error('Exactly one parameter must be specified for plotting');
            end
            plotIdx = plotIdx(1);
            
        
            data = obj.params(1:obj.numGroups,:);
            ipdata = data(:,params ~= -1);
            fixparam = params(:,params ~= -1);
            rowIndex = obj.findLike(ipdata,fixparam);

            plotparam = data(rowIndex,plotIdx);
            plotm = obj.metrics(rowIndex, metricIdx);
            
            if isempty(plotparam)
                warning('No matching parameter sets found');
                cprintf('yellow','No matching parameter sets found\n');
                p = Inf;m=[];
                return;
            end
            
            [p, sortOrder] = sort(plotparam);
            m = plotm(sortOrder);
            

            figure;
            plot(p, m, 'Marker', '.','MarkerSize',1 ,'LineWidth', 1.2);
            xlabel(['Param ' num2str(plotIdx)]);
            ylabel(['Metric ' num2str(metricIdx)]);
            xlim([min(p), max(p)]);
            title(['Effect of Param ' num2str(plotIdx) ' on Metric ' num2str(metricIdx)]);
            % grid on;
        end
        
        function idx = findParam(obj, param)
            if obj.numGroups == 0
                idx = 0;
                return
            end
            % diff_matrix = abs(obj.params(1:obj.numGroups, :) - param);
            % if ~(sum(matchIdx) ~= 1 || sum(matchIdx) ~= 0)
            % end
            % if any(matchIdx)
            % else
            % end
            [~, idx] = ismember(param, obj.params, 'rows');
        end

        function exists = checkExist(obj, nparam)
            if obj.numGroups == 0
                exists = 0;
                return
            end
          
            [~, ia] = ismember(nparam, obj.params, 'rows');
            exists = all(ia > 0);
        end

        % function exists = checkExist(obj, nparam)
        %     exists = true;
        %     for pp=1:size(nparam,1)
        %         exists = exists & findParam(obj,nparam(pp,:));
        %         if ~exists 
        %             break
        %         end
        %     end
        % end
       
        function pToFile(obj,filename, metricIdx)
            if nargin < 3
                metricIdx = 1;
            end
            if metricIdx < 1 || metricIdx > 6
                error('Invalid metric index');
            end
            
            param = obj.params(1:obj.numGroups,:);
            metric = obj.metrics(1:obj.numGroups, metricIdx);
            
            data = [param, metric];
            
            [sortedData, ~] = sortrows(data);

            [parentDir, ~, ~] = fileparts(filename);
            if ~exist(parentDir, 'dir')
                mkdir(parentDir);
            end

            fid = fopen(filename, 'w');
            if fid == -1
                error('无法打开文件');
            end
            
            fprintf(fid, 'Parameters and Metrics，Seeds: %s\n',num2str(obj.seeds));
            fprintf(fid, '-----------------------------------------------\n');
            
            for i = 1:size(sortedData, 1)
                paramStr = sprintf([repmat('%-10g', 1, size(param, 2)), '%-10g'], sortedData(i, 1:end-1), sortedData(i, end));
                fprintf(fid, '%s\n', paramStr);
            end
            
            fclose(fid);
            fprintf('数据已保存到文件: %s\n', filename);
        end

        function save(obj, filename)
            [parentDir, ~, ~] = fileparts(filename);
            if ~exist(parentDir, 'dir')
                mkdir(parentDir);
            end

            save(filename, 'obj');
            fprintf('已将模型保存到文件: %s\n', filename);
        end
        
        function obj = load(~,filename)
            data = load(filename);
            obj = data.obj;
            fprintf('已从 MAT 文件加载数据: %s\n', filename);
        end

    end
end
