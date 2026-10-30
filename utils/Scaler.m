classdef Scaler
    % X ：n*d
    properties
        meanX
        stdX
        epsilon = 1e-8
        dim = 1;
    end
    
    methods
        function obj = Scaler(dim)
            if nargin > 0
                obj.dim = dim;
            end
        end
        
        function obj = fit(obj, X)
            obj.meanX = mean(X, obj.dim); % 1*d
            obj.stdX = std(X, 0, obj.dim);
            % zero_col = (stdX == 0);
            % stdX(zero_col) = obj.epsilon;
            % obj.stdX = stdX;
        end
        
        function X_scaled = transform(obj, X)
            X_scaled = (X - obj.meanX) ./ (obj.stdX + obj.epsilon);
        end
        
        function [X_scaled,obj] = fit_transform(obj, X)
            obj = obj.fit(X);
            X_scaled = obj.transform(X);
        end
    end
end
