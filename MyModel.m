classdef MyModel
    properties
        name = 'MyModel';
        max_iter = 200;
        iter_tol = 1e-3;
        epsilon_w = 1e-10;
        epsilon_gamma = 1e-10;
        epsilon_D = 1e-10;
         
        verbose = 0;
        keyboard = 1;
        Loss = [];
        AugLoss = [];
        Err = [];
        Err2 = [];
        ResidualDetail = [];
        LossRate = [];
        AugLossRate = [];
        RateSource = 'AugLoss';
        StatErr = [];
        StationarityDetail = [];
        ComputeStationarity = true;
        Converged = false;
        Iterations = 0;
        ZStep = [];
        History_A;
        History_S;
        History_B;


        start_time=1;
        runtime = 0;
        
        alpha;
        beta;
        gamma = [];
        yita;
        lambda;
        p;

       
        
        n;
        M;
        d;
        c;
        
        W;
        A;
        P;
        Z;
        B;
        S;
        A_tensor;
        J_tensor;

        
        La;
        Lt;
        T;
        
      



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

        function obj = setParams(obj, params)
            obj.yita = params(1);
            obj.lambda = params(2);
            obj.alpha = params(3); 
            obj.beta = params(4);
            obj.p = params(5);
            
        end

        % Initialize
        function obj = init(obj,n, c, M, d, X)
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
            obj.W = cell(M, 1);
            obj.A = cell(M, 1);
            obj.P = cell(M, 1);
            obj.B = cell(M, 1);
            obj.S = cell(M, 1);
            obj.La = cell(M,1);
            obj.Lt = cell(M,1);
            obj.T = cell(M,1);
            obj.A_tensor = zeros(n, n, M);
            obj.J_tensor = zeros(n, n, M);
            obj.Q_tensor = zeros(n, n, M);
            obj.Q1 = cell(M, 1);
            obj.Q2 = cell(M, 1);
            obj.Q3 = cell(M, 1);

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


            for m = 1:M
                obj.W{m} = zeros(d(m),c);
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

        function [f,df] = Fun(obj, X, Y, fTNN, HX, PS)

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
                f4=f4+g(m)*(graph_difference+obj.epsilon_gamma) ...
                    +1/(4*g(m));
                f4_aug=f4_aug+g(m)*graph_difference;
                f5=f5+sum(sqrt(sum(Wm.^2,2)));
                f6=f6+sum(sqrt(sum(Bm.^2,2)));

                residual1=Am-Pm;
                if nargin >= 6 && ~isempty(PS)
                    PSm=PS{m};
                else
                    PSm=Pm*Sm;
                end
                residual2=Sm-PSm-Bm;
                residual4=Sm-HX{m};
                f8=f8+mu/2*norm(residual1+Q1m/mu,'fro')^2;
                f9=f9+mu/2*norm(residual2+Q2m/mu,'fro')^2;
                f11=f11+mu/2*norm(residual4+Q3m/mu,'fro')^2;
                
            end

            
            if nargin > 3 && ~isempty(fTNN)
                f7=fTNN;
            else
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
            
            [n, c, M, d] = getDataInfo(X,Y);



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
