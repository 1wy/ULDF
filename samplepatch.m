function [patchmatrix, patchlabels] = samplepatch(InImg, ImgLabel, ImgSize, PatchSize, StepSize) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% ImgLabel         Labels of InImg (column vector)
% ImgSize          Size of images
% PatchSize        Size of patches
% StepSize         Size of step
% =======OUTPUT============
% patchmatrix       a patch matrix, each row is a sample 
% patchlabels       the labels of patches

addpath('./Utils')
numImg = length(ImgLabel);
maxnumpatch = (floor((ImgSize(1)-PatchSize)/StepSize)+1)...  %max number of patches can be extracted
              *(floor((ImgSize(2)-PatchSize)/StepSize)+1);
maxnumneed =200; 
maxnumpatch = min(maxnumpatch, maxnumneed);
patchmatrix = zeros(numImg*maxnumpatch,PatchSize^2);
patchlabels = zeros(numImg*maxnumpatch,1);

for i = 1:numImg
    im = im2col_general(InImg{i},[PatchSize PatchSize],[StepSize StepSize]); % collect all the patches of the ith image in a matrix
%     im = bsxfun(@minus, im, mean(im)); % patch-mean removal
    idx = randperm(size(im,2));
    patchmatrix(1+(i-1)*maxnumpatch:i*maxnumpatch,:) = im(:,idx(1:maxnumpatch))';
    patchlabels(1+(i-1)*maxnumpatch:i*maxnumpatch,:) = ImgLabel(i);
end

%%