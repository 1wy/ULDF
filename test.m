function [acc,proestimatetest] = test(models,para)
%This is a function that tests the model.
%It includes encoding using fisher encoding with SPM
%and train a svm classifier.
numcode = para.numcode;
data_dir = para.data_dir;
ImgSize = para.ImgSize;
PatchSize = para.PatchSize;
StepSize = para.StepSize;
pyramid = para.pyramid;
knn = para.knn;

binnum = sum(pyramid.^2);
fea = zeros(10000,2*numcode*para.numopen*binnum,'single');
labels = zeros(10000,1);
cnt = 1;
for classi = 0:9
    load(['data' num2str(ImgSize(1)) '/test' num2str(classi)]);
    D = D/255;
    Img = mat2imgcell(D',ImgSize(1),ImgSize(2),'gray');
    numImg = length(Img);
    ImgLabel = classi*ones(numImg,1);
    fea(cnt:cnt+numImg-1,:) = fisher_output(Img,para);
    labels(cnt:cnt+numImg-1) = ImgLabel;
    cnt = cnt + numImg;
end

fea = sparse(double(fea));
labels = labels;
% save(['featuredata/test'],'fea','labels','-v7.3');
[Label_pre, acc, proestimatetest] = predict(labels,fea, models); % label predictoin by libsvm
save([data_dir '/proestimatetest'],'proestimatetest');
fprintf('the accuracy is %d \n',acc);
end

