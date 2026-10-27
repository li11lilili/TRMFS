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
pre_path = 'ADNI\ADNINCLMCI';
[X,Y] = getADData(DataPath,Group,15,false);



[n, c, M, d] = getDataInfo(X,Y);
parts = strsplit(pre_path, '\'); 
task = parts{2}; 
distribution(Y);

cp = 1;
save = 1;  
if cp == 0
    save = 0;
end
parrun = 1; 
dep = [];
method = 'MyModel';
params = MyModel().init_param();
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

pool = gcp();

possibleFiles = {
    'svmtrain.m', 'svmpredict.m', ...
    'svmtrain.mexw64', 'svmpredict.mexw64', ...
    'libsvmread.m', 'libsvmwrite.m'
};

fullFiles = fullfile(libsvmPath, possibleFiles);

existsMask = cellfun(@(f) exist(f, 'file') == 2, fullFiles);
attachedFiles = fullFiles(existsMask);

addAttachedFiles(pool, attachedFiles);
outpath = 'output'; 
if exist([outpath,'\',pre_path,'\',method,'_re.mat'],'file') && save
    recorder = load([outpath,'\',pre_path,'\',method,'_re.mat']).obj;
else 
    recorder = [];
end
s_time = tic;


recorder = runComputationSVM(np,method,X,Y,recorder,cp,parrun,task,pre_path,outpath,dep);

seeds = [224 927 283 120 1486 491 666 5540 3654 167];
r = 10;
k = 10;

recorder.printMetrics()
elapsed_time = toc(s_time);
recorder.plotROC(1);
recorder.plotLoss(1);
recorder.plotConvergence(1,1);

[acc,stdAcc] = recorder.getM('acc');
fprintf('runtime: %.2fs.   %s \tacc=%.4f(±%.4f)\n', ...
    elapsed_time,mat2str([recorder.nfs, recorder.params]),acc,stdAcc);

optparams = recorder.params; 

pp = [-1, optparams];
pt = recorder.pt;
pt.plotEffect(pp); 

pp(2) = -1;
pt.plotBar(pp);

