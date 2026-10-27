clc;
clear;
close all;

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir,'filter'));
addpath(fullfile(rootDir,'update'));
addpath(fullfile(rootDir,'utils'));
addpath(fullfile(rootDir,'MK_SVM'));


DataPath = {
    'Datasets\ADNI\ADNI_combined_DATA_MRI';
    'Datasets\ADNI\ADNI_combined_DATA_PET'
};

Group = {'LMCI','CN'};
pre_path = 'ADNI\ADNINCLMCIiter200';
[X,Y] = getADData(DataPath,Group,15,false);



[n, c, M, d] = getDataInfo(X,Y);
parts = strsplit(pre_path, '\'); % 按 '\' 分割字符串
task = parts{2}; % 提取 '\' 前面的部分（AC）
distribution(Y);
%%
cp = 1; % =1表示记录计算过程
save = 1;  
if cp == 0
    save = 0;
end
parrun = 1; % =1表示开启并行池 
dep = [];
% param = [1, 10];
method = 'MyModel';
% params = MyModel().init_param([100,0.1,10,0.08,13,0.57,0.012]);
params = MyModel().init_param();
% My().setParams([10,1,0.19,4,0.599,0.015]);
% params=[10,1,0.19,4,0.599,0.015];
np = 5;
libsvmPath = getenv('LIBSVM_MATLAB_PATH');
if ~isempty(libsvmPath)
    addpath(libsvmPath);
elseif ~isempty(which('svmtrain'))
    libsvmPath = fileparts(which('svmtrain'));
else
    error(['LIBSVM was not found. Add its MATLAB directory to the path ', ...
        'or set the LIBSVM_MATLAB_PATH environment variable.']);
end

% % 初始化并行环境
% addpath(genpath('D:\SLEP\SLEP_package_4.1'));
% if isempty(gcp('nocreate'))
%     pool = parpool;
%     pctRunOnAll addpath(genpath('D:\SLEP\SLEP_package_4.1'));
%     addAttachedFiles(pool, {
%         'D:\SLEP\SLEP_package_4.1\SLEP\functions\L1\L1R\nnLeastR.m',
%         'D:\SLEP\SLEP_package_4.1\SLEP\functions\mex\*.mexw64'
%     });
% end


% 获取当前并行池
%pool = gcp();

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(12); % 根据当前24核CPU的实测结果开启10个worker
elseif pool.NumWorkers ~= 12
    delete(pool);       % 关闭原有并行池
    pool = parpool(12); % 重新开启新的并行池，设置为10个worker
end

% 定义可能的文件（.m 和 .mexw64）
possibleFiles = {
    'svmtrain.m', 'svmpredict.m', ...
    'svmtrain.mexw64', 'svmpredict.mexw64', ...
    'libsvmread.m', 'libsvmwrite.m'
};

% 拼接完整路径
fullFiles = fullfile(libsvmPath, possibleFiles);

% 检查哪些文件存在
existsMask = cellfun(@(f) exist(f, 'file') == 2, fullFiles);
attachedFiles = fullFiles(existsMask);

% 附加存在的文件
addAttachedFiles(pool, attachedFiles);
% 检查这些文件是否存在，并将存在的文件附加到并行池中，确保并行任务中也能调用这些文件


%%
outpath = 'output'; % 定义结果输出目录为 'output' 文件夹（用于保存结果文件）
if exist([outpath,'\',pre_path,'\',method,'_re.mat'],'file') && save
    recorder = load([outpath,'\',pre_path,'\',method,'_re.mat']).obj;
else 
    recorder = [];
end
s_time = tic;

%%
% % 调整两个参数
recorder = runComputationSVM(np,method,X,Y,recorder,cp,parrun,task,pre_path,outpath,dep);

%%
% 运行指定参数，指定 10 个随机种子，用于 10 次重复运行（r=10），每次交叉验证的折数为 10（即 10 折交叉验证）
seeds = [224 927 283 120 1486 491 666 5540 3654 167];
r = 10;
k = 10;
% cp:是否使用 checkpoint,rp：是否重新训练，pr：是否并行运行，task：任务名
%recorder = krun(method,X,Y,params,r,k,'s',seeds,'cp',cp,'rp',1,'recorder',recorder,'pr',parrun,'task',task);


recorder.printMetrics()
elapsed_time = toc(s_time);
recorder.plotROC(1);
recorder.plotLoss(1);
recorder.plotConvergence(1,1);
% 计算整个流程的运行时间，获取准确率及标准差，并输出
[acc,stdAcc] = recorder.getM('acc');
fprintf('runtime: %.2fs.   %s \tacc=%.4f(±%.4f)\n', ...
    elapsed_time,mat2str([recorder.nfs, recorder.params]),acc,stdAcc);

optparams = recorder.params; % 获取训练后最优参数

% num of feature
pp = [-1, optparams];
pt = recorder.pt;
pt.plotEffect(pp); % 绘制最优参数的柱状图

pp(2) = -1;
pt.plotBar(pp);

