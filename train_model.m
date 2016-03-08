
function[ models ] = train_model(para)
%This is a function that trains the model.
%It includes learning a codebook using DMM,
%encoding using fisher encoding with SPM,
%and train a svm classifier.
numcode = para.numcode;
data_dir = para.data_dir;
ImgSize = para.ImgSize;
PatchSize = para.PatchSize;
StepSize = para.StepSize;
pyramid = para.pyramid;



load([data_dir '/digit0']);
D_dim1 = size(D,1);
D_dim2 = size(D,2);
data = zeros(D_dim1*10,D_dim2);

for classi = 0:9
    load([data_dir '/digit' num2str(classi)]);
%     data = [data;D];   
    data(classi*D_dim1+1:(classi+1)*D_dim1,:) = D;
end
data = data/255;
%%===========================reduce dimention==============================
load([data_dir '/' 'mnist_weights']);
addpath vlfeat\vlfeat-0.9.18\toolbox\;
vl_setup;

N = size(data,1);
data = [data ones(N,1)];
w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
w4probs = (w3probs*w4)';
clear data;
%%===========================get dictionary================================
% fprintf('\n ====== getting codebook ======= \n');
% [means, covariances, priors] = vl_gmm(w4probs, numcode);
% save([data_dir '/' 'dictionary'],'means','covariances','priors'); 
% fprintf('\n ====== finish getting codebook ======= \n');
%%=========================================================================
binnum = sum(pyramid.^2);
fea = zeros(60000,2*numcode*para.numopen*binnum,'single');
labels = zeros(60000,1);
cnt = 1;
for classi = 0:9
    load(['data' num2str(ImgSize(1)) '/digit' num2str(classi)]);
    D = D/255;
    Img = mat2imgcell(D',ImgSize(1),ImgSize(2),'gray');
    numImg = length(Img);
    ImgLabel = classi*ones(numImg,1);
%     encoding using fisher encoding with SPM
    fea(cnt:cnt+numImg-1,:) = fisher_output(Img,para);
    labels(cnt:cnt+numImg-1) = ImgLabel;
    cnt = cnt + numImg;
end
fprintf('\n ====== Training Linear SVM Classifier ======= \n')

labelstra = single(labels);
tic;
models = train(labelstra, fea, '-s 1 -c 1 -q'); % we use linear SVM classifier (C = 1), calling liblinear library
LinearSVM_TrnTime = toc;

for i = 1:6
[Label_pre, acc(i),proestimattra(1+(i-1)*10000:i*10000,:)] = predict(double(labelstra(1+(i-1)*10000:i*10000)),...
                                sparse(double(fea(1+(i-1)*10000:i*10000,:))), models);

end
save([data_dir '/proestimattra'],'proestimattra');
accuracytrain = mean(acc); 
end


