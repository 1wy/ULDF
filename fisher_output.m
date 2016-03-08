function [fea] = fisher_output(InImg, para)
% This is a function that ecoding using fisher encoding with SPM
%Parameters:
%   1.InImg -- N x 1 cells. Each cell is an image.
%   2.para -- Parameters include numcode, data_dir and so on.
%Return value:
%   fea -- N x K matrix. Each row is a feature vector of an image. 
    
numcode = para.numcode;
data_dir = para.data_dir;
ImgSize = para.ImgSize;
PatchSize = para.PatchSize;
StepSize = para.StepSize;
pyramid = para.pyramid;


addpath('./Utils');
load([data_dir '/' 'mnist_weights']);
load([data_dir '/' 'dictionary']);

binnum = sum(pyramid.^2);
numImg = length(InImg);
fea = zeros(numImg,2*numcode*para.numopen*binnum,'single');

% cnt = 0;
for i = 1:numImg 
    
    img = InImg{i};
    [imgheight, imgwidth] = size(img);         
    im = im2col_general(img,[PatchSize PatchSize], [StepSize StepSize])'; % collect all the patches of the ith image in a matrix
    N = size(im,1);
    im = [im ones(N,1)];
    w1probs = 1./(1 + exp(-im*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
    w4probs = (w3probs*w4)';
      
    feaSet.feaArr = w4probs;   
    nrow = floor((imgwidth-PatchSize)/StepSize)+1;
    ncol = floor((imgheight-PatchSize)/StepSize)+1;
    
    feaSet.width = nrow;
    feaSet.height = ncol;
    

    [feaSet.x feaSet.y] = meshgrid(1:nrow,1:ncol);
    feaSet.x = feaSet.x(:);
    feaSet.y = feaSet.y(:);
    
    fea(i,:) = single(pooling(feaSet, pyramid, data_dir));
    %feaSet.feaArr:dim*n,each column is a sample
    %V:dim*n, each column is a code   

    if 0==mod(i,100);display(['Finish outputing ' num2str(i) 'th img at current stage...']);end
end

