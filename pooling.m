% ========================================================================
% Fisher encoding and pooling to form the image feature
% USAGE: [beta] = pooling(feaSet, pyramid, data_dir)
% Inputs
%       feaSet      -the coordinated local descriptors
%       pyramid     -the spatial pyramid structure
%       data_dir         -the directory containing the dictionary
% Outputs
%       beta        -the output image feature
%
% Written by Yang Wang
% ========================================================================

function [beta] = pooling(feaSet, pyramid, data_dir)

load([data_dir '/' 'dictionary']);
dSize = 2*size(means,1)*size(means,2);
nSmp = size(feaSet.feaArr, 2);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end     
        tempfea = vl_fisher(feaSet.feaArr(:,sidxBin), means, covariances, priors);
        beta(:, bId) = tempfea/norm(tempfea);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:)';
beta = beta/norm(beta);
