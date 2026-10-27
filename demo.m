clc;
clear;
close all;

% Synthetic-data demonstration of the core TRMFS optimization algorithm.
rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir,'filter'));
addpath(fullfile(rootDir,'update'));
addpath(fullfile(rootDir,'utils'));

rng(42);
n = 40;                 % Number of samples
c = 2;                  % Number of classes
M = 2;                  % Number of modalities
d = [12, 10];           % Feature dimensions

X = cell(1,M);
X{1} = randn(n,d(1));
X{2} = randn(n,d(2));

labels = [ones(n/2,1); 2*ones(n/2,1)];
Y = zeros(n,c);
Y(sub2ind([n,c],(1:n)',labels)) = 1;

% Parameter order: [eta, lambda, alpha, beta, p].
params = [1, 1, 1, 1, 2];
seed = 42;

model = MyModel();
model.verbose = 0;
model.ComputeStationarity = false;
model = model.run(X,Y,params,seed);

fprintf('TRMFS finished after %d iterations. Converged: %d\n', ...
    model.Iterations,model.Converged);
fprintf('Final AugLoss rate: %.6g\n',model.AugLossRate(end));
fprintf('Final maximum primal residual: %.6g\n',model.Err2(end));

for m = 1:M
    fprintf('W{%d} size: %d x %d\n',m,size(model.W{m},1),size(model.W{m},2));
end
