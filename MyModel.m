classdef MyModel
    properties
        % 基本参数
        name = 'MyModel';
        max_iter = 200;
        iter_tol = 1e-3; %迭代容忍误差
        epsilon_w = 1e-10;
        epsilon_gamma = 1e-10;
        epsilon_D = 1e-10;
         
        verbose = 0; % 是否打印调试信息
        keyboard = 1;% 是否使用调试模式
        Loss = [];   % 记录原目标函数值
        AugLoss = []; % 记录乘子更新前的增广拉格朗日函数值
        Err = [];       % 当前选定的函数相对变化率
        Err2 = [];      % max(r1,r2,r3,r4)，用于停止判断
        ResidualDetail = []; % 4×K，每列保存当轮 [r1;r2;r3;r4]
        LossRate = [];      % 原目标函数相对变化率
        AugLossRate = [];   % 增广拉格朗日函数相对变化率
        RateSource = 'AugLoss'; % 式(11)的增广目标函数变化率用于停止判断
        StatErr = [];
        StationarityDetail = [];
        ComputeStationarity = true;
        Converged = false;
        Iterations = 0;
        ZStep = [];
        History_A;
        History_S;
        History_B;


        start_time=1; %计时器起点
        runtime = 0;  %运行时间
        
        % 超参数
        alpha;   % 样本流行项
        beta;    % 特征流行项
        gamma = [];   % 自表达矩阵学习共识的系数
        yita;    % W的稀疏
        lambda;  % B的稀疏
        p;       % 图过滤阶数

       
        
        % 数据维度
        n;  % 样本数
        M;  % 模态数
        d;  % 各模态特征维度[] 1行M列向量
        c;  % 目标类别数
        
        % 优化变量
        W;  % 各模态投影矩阵 cell
        A;  % 各模态相似矩阵 cell  
        P;  % A的辅助副本
        Z;  % 共识图 n*n
        B;  % 各模态自表达误差矩阵 cell  n*d     
        S;  % 各模态平滑表示 cell n*d
        A_tensor; % Am组成的张量,表示沿着第三维度把 cell 里的所有矩阵拼接起来
        J_tensor; % 辅助变量替代A_tensor

        
        % 样本拉普拉斯
        La;
        % 特征拉普拉斯
        Lt;
        % 特征相似矩阵 cell
        T;
        
      



        % 拉格朗日乘子
        Q_tensor;
        Q1;
        Q2;
        Q3;
        mu=1;          % ADMM penalty initial value (rho in LRTLS)
        mu_max = 1e6; % upper bound of the ADMM penalty
        delta = 1.1;  % penalty growth factor, 1 <= delta < 2

    end

    methods
        function obj = MyModel()
        end

        % Set parameters 总流程里的输入参数
        function obj = setParams(obj, params) %传入 params 数组，依次设置超参数beta、gamma            
            obj.yita = params(1);
            obj.lambda = params(2);
            obj.alpha = params(3); 
            obj.beta = params(4);
            obj.p = params(5);
            
        end

        % Initialize
        function obj = init(obj,n, c, M, d, X) %样本数、类别数、模态数、维度
            obj.M = M;
            obj.n = n;
            obj.d = d;
            obj.c = c; 
            obj.Loss = [];
            obj.AugLoss = [];
            obj.Err = [];
            obj.Err2 = [];
            obj.ResidualDetail = [];
            obj.LossRate = [];
            obj.AugLossRate = [];
            obj.StatErr = [];
            obj.StationarityDetail = [];
            obj.Converged = false;
            obj.Iterations = 0;
            obj.ZStep = [];
            obj.W = cell(M, 1); % 存储每个模态的投影矩阵
            obj.A = cell(M, 1); % 存储每个模态的相似矩阵（自表达）
            obj.P = cell(M, 1);
            obj.B = cell(M, 1); % 存储每个模态的误差矩阵
            obj.S = cell(M, 1); % 存储每个模态的平滑表示
            obj.La = cell(M,1); % 存储每个Am的拉普拉斯La
            obj.Lt = cell(M,1); % 存储特征的拉普拉斯Lt
            obj.T = cell(M,1);  % 特征相似矩阵
            obj.A_tensor = zeros(n, n, M);
            obj.J_tensor = zeros(n, n, M);
            obj.Q_tensor = zeros(n, n, M);
            obj.Q1 = cell(M, 1);
            obj.Q2 = cell(M, 1);
            obj.Q3 = cell(M, 1);

            % 从每个原始模态中学习一个预定义样本图。
            % 初始化过程不使用类别标签，仍然保持无监督学习。
            use_data_graph = nargin >= 6 && ~isempty(X);
            if use_data_graph
                A_initial = getKSimilarity(X,[],false,[]);
                for m = 1:M
                    Am = max(real((A_initial{m}+A_initial{m}')/2),0);
                    Am(1:n+1:end) = 0;
                    obj.A{m} = Am;
                end

                obj.Z = obj.A{1};
                for m = 2:M
                    obj.Z = obj.Z+obj.A{m};
                end
                obj.Z = obj.Z/M;
            end

            obj.History_A=cell(obj.max_iter,1);
            obj.History_S=cell(obj.max_iter,1);
            obj.History_B=cell(obj.max_iter,1);


            for m = 1:M % 对每个模态初始化
                obj.W{m} = zeros(d(m),c); % 初始化每个模态的投影矩阵
                if ~use_data_graph
                    obj.A{m} = zeros(n, n);
                end
                obj.P{m} = obj.A{m};
                obj.T{m} = zeros(d(m),d(m));
                
                obj.B{m} = zeros(n,d(m));
                obj.La{m} = zeros(n,n);
                obj.Lt{m} = zeros(d(m),d(m));
                obj.A_tensor(:,:,m) = obj.A{m};
                obj.Q1{m} = zeros(n,n);
                obj.Q2{m} = zeros(n,d(m));
                obj.Q3{m} = zeros(n,d(m));
            end
            if ~use_data_graph
                obj.Z = zeros(n,n);
            end
            obj.J_tensor = obj.A_tensor;
            obj.Q_tensor = zeros(n,n,M);
            obj.S = graphFilterS(obj,X);

            obj.start_time = tic;
        end

        function [f,df] = Fun(obj, X, Y, fTNN, HX, PS) %计算增广拉格朗日函数

            a=obj.alpha;
            b=obj.beta;            
            g=obj.gamma;
            yi=obj.yita;
            l=obj.lambda;
            mu=obj.mu;

            if nargin < 5 || isempty(HX)
                HX=graphFilterS(obj,X);
            end


            f1=0;
            f2=0;
            f3=0;
            f4=0;
            f4_aug=0;
            f5=0;
            f6=0;
            f8=0;
            f9=0;
            f11=0;

            for m=1:obj.M
                Sm=obj.S{m};
                Wm=obj.W{m};
                Bm=obj.B{m};
                Am=obj.A{m};
                Pm=obj.P{m};
                Q1m=obj.Q1{m};
                Q2m=obj.Q2{m};
                Q3m=obj.Q3{m};
                
                Ltm=obj.Lt{m};
                Lam=obj.La{m};

                XW=Sm*Wm;
                
                f1=f1+norm(Y-XW,'fro')^2;                 
                f2=f2+trace(XW'*Lam*XW);
                f3=f3+trace(Wm'*Ltm*Wm);
                graph_difference=norm(Am-obj.Z,'fro')^2;
                % 保留修改前 Loss 中的半二次等价形式。
                f4=f4+g(m)*(graph_difference+obj.epsilon_gamma) ...
                    +1/(4*g(m));
                % 论文式(11)中的加权共识图项。
                f4_aug=f4_aug+g(m)*graph_difference;
                f5=f5+sum(sqrt(sum(Wm.^2,2))); %2,1范数
                f6=f6+sum(sqrt(sum(Bm.^2,2)));

                residual1=Am-Pm;
                if nargin >= 6 && ~isempty(PS)
                    PSm=PS{m};
                else
                    PSm=Pm*Sm;
                end
                residual2=Sm-PSm-Bm;
                residual4=Sm-HX{m};
                % 式(11)采用的平方配方形式
                f8=f8+mu/2*norm(residual1+Q1m/mu,'fro')^2;
                f9=f9+mu/2*norm(residual2+Q2m/mu,'fro')^2;
                f11=f11+mu/2*norm(residual4+Q3m/mu,'fro')^2;
                
            end

            
            if nargin > 3 && ~isempty(fTNN)
                f7=fTNN;
            else
                % 初始化阶段没有现成的 fTNN，此时直接计算 J 的归一化 TNN
                J_f = fft(obj.J_tensor, [], 3);
                n3 = size(obj.J_tensor,3);
                val = 0;
                for q = 1:n3
                    Fq = J_f(:, :, q);
                    s = svd(Fq, 'econ');
                    val = val + sum(s);
                end
                f7=val/n3;
            end

            residual3=obj.A_tensor-obj.J_tensor;
            f10=mu/2*norm(residual3+obj.Q_tensor/mu,'fro')^2;

            df=f1+a*f2+b*f3+f4+yi*f5+l*f6+f7;
            augmented_objective=f1+a*f2+b*f3+f4_aug+yi*f5+l*f6+f7;
            f=augmented_objective+f8+f9+f10+f11;
%             fprintf('f1=%.4f, f2=%.4f, f3=%.4f, f4=%.4f, f5=%.4f, f6=%.4f, f7=%.4f, total=%.4f\n', f1, f2, f3, f4, f5, f6, f7, f);

        end


        function obj = run(obj,X,Y,param,seed)
            warning('off','all');

            rate_source=validatestring(obj.RateSource,{'Loss','AugLoss'});
            obj.RateSource=rate_source;

            if obj.mu <= 0 || obj.mu > obj.mu_max
                error('MyModel:InvalidMu', ...
                    'The initial penalty mu must satisfy 0 < mu <= mu_max.');
            end
            if obj.max_iter < 100 || obj.max_iter > 200 || ...
                    obj.max_iter ~= floor(obj.max_iter)
                error('MyModel:InvalidMaxIter', ...
                    'max_iter must be an integer within [100, 200].');
            end
            
            [n, c, M, d] = getDataInfo(X,Y); %调用外部函数 获取信息



            obj = setParams(obj,param);
            % rng(seed);
            obj = obj.init(n, c, M, d, X);

            obj.T = getSimilarityT(obj,X);
            obj.Lt=getLaplacianT(obj);
            obj.La=getLaplacianA(obj);
            obj.gamma=ones(M, 1);
            obj.gamma=updateGamma(obj);
            

            HX=obj.S;
            [augmented_old,objective_old] = obj.Fun(X,Y,[],HX);
            obj.Loss(1)=objective_old;
            obj.AugLoss(1)=augmented_old;

            converged=false;
            for iter=1:obj.max_iter
                obj.W=updateW(obj,obj.S,Y);
                obj.P=updateP(obj,obj.S);
                obj.A=updateA(obj,obj.S);
                obj.La=getLaplacianA(obj);
                obj.B=updateB_L21(obj,obj.S);

                for m=1:obj.M
                    obj.A_tensor(:,:,m)=obj.A{m};
                end
                [obj.J_tensor,fTNN]=updateJtensor(obj);
                [obj.S,s_linear_residual]=updateS(obj,X,Y,HX);
                [obj.Z,z_stationarity,HX,z_step]=updateZ(obj,X);
                obj.ZStep=z_step;
                obj.gamma=updateGamma(obj);

                PS=cell(obj.M,1);
                for m=1:obj.M
                    PS{m}=obj.P{m}*obj.S{m};
                end
                [residual,residual_detail]=computePrimalResidual(obj,X,HX,PS);

                [obj.Q1,obj.Q2,obj.Q3,obj.Q_tensor]= ...
                    updateLagrangian(obj,HX,PS);
                obj.mu=min(obj.mu_max,obj.delta*obj.mu);

                % 原目标函数仍记录为 Loss；式(11)记录为 AugLoss。
                % 乘子和惩罚参数更新后再计算式(11)，与一次完整迭代状态对应。
                [augmented_value,objective]=obj.Fun(X,Y,fTNN,HX,PS);
                loss_rate=abs(objective-objective_old) ...
                    /max(abs(objective_old),eps);
                augloss_rate=abs(augmented_value-augmented_old) ...
                    /max(abs(augmented_old),eps);
                if strcmp(obj.RateSource,'Loss')
                    rate=loss_rate;
                else
                    rate=augloss_rate;
                end
                if obj.ComputeStationarity
                    [stationarity,stationarity_detail]= ...
                        computeStationarityResidual(obj,Y, ...
                        z_stationarity,s_linear_residual);
                else
                    stationarity=NaN;
                    stationarity_detail=[];
                end

                obj.Loss(end+1)=objective;
                obj.AugLoss(end+1)=augmented_value;
                obj.LossRate(end+1)=loss_rate;
                obj.AugLossRate(end+1)=augloss_rate;
                obj.Err(end+1)=rate;
                obj.Err2(end+1)=residual;
                if obj.ComputeStationarity
                    obj.StatErr(end+1)=stationarity;
                    obj.StationarityDetail(:,end+1)=stationarity_detail(:);
                end
                obj.ResidualDetail(:,end+1)=residual_detail(:);

                if obj.verbose
                    fprintf(['Iter %d/%d | rate(%s)=%g ', ...
                        '[Loss=%g AugLoss=%g] | ', ...
                        'residual=%g [r1=%g r2=%g r3=%g r4=%g] | ', ...
                        'stationarity=%g | mu=%g\n'], ...
                        iter,obj.max_iter,obj.RateSource,rate,loss_rate, ...
                        augloss_rate,residual,residual_detail(1), ...
                        residual_detail(2),residual_detail(3), ...
                        residual_detail(4),stationarity,obj.mu);
                end

                if residual < obj.iter_tol && rate < obj.iter_tol
                    converged=true;
                    break;
                end

                objective_old=objective;
                augmented_old=augmented_value;
            end

            obj.Converged=converged;
            obj.Iterations=iter;
            if ~converged
                warning('MyModel:NotConverged', ...
                    ['The unified optimization did not converge in %d ', ...
                    'iterations (rate=%g, residual=%g, stationarity=%g).'], ...
                    obj.max_iter,rate,residual,stationarity);
            end
            if obj.verbose
                for i = 1:length(obj.Loss)
                    fprintf('Loss{%d} = %.6f\n', i, obj.Loss(i));
                end
            end
        end
        



        function parameter = init_param(~,fix)
            if nargin < 2
                fix = [Inf,Inf,Inf,Inf,Inf];
            end


            
            yitaSpace=logspace(-5,5,11);
            lambdaSpace=logspace(-5,5,11);
            alphaSpace = logspace(-5,5,11);
            betaSpace = logspace(-5,5,11);
            pSpace=[1,2,3,4,5];
            
           
            
            if fix(1) ~= Inf
                yitaSpace = fix(1);
            end
            if fix(2) ~= Inf
                lambdaSpace = fix(2);
            end  
            if fix(3) ~= Inf
                alphaSpace = fix(3);
            end
            if fix(4) ~= Inf
                betaSpace = fix(4);
            end
            if fix(5) ~= Inf
                pSpace = fix(5);
            end
            
              
            

            paramSpace = {yitaSpace,lambdaSpace,alphaSpace,betaSpace,pSpace};
            parameter = combvec(paramSpace{:})';
            parameter = sortrows(parameter, 'ascend');
        end
    end
end
