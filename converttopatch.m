
function converttopatch(para)
% This is a function that collects patches and select some randomly for training  

StepSize = 2; 
ImgSize = para.ImgSize;
PatchSize = para.PatchSize;
data_dir = para.data_dir;

for classi = 0:9
    load(['data' num2str(ImgSize(1)) '/digit' num2str(classi)]);
    Img = mat2imgcell(D',ImgSize(1),ImgSize(2),'gray');
    numImg = length(Img);
    [patchmatrix, trapatchlabels] = samplepatch(Img, classi*ones(numImg,1), ImgSize, PatchSize, StepSize);
    rand('state',0);
    randidx = randperm(size(patchmatrix,1));
    % we randomly select 20000 patches from each class for training
    % autoencoder
    if length(randidx) > 20000
        patchmatrix = patchmatrix(randidx(1:20000),:);
    end
    D = patchmatrix;
    save([data_dir '/' 'digit' num2str(classi) '.mat'],'D');
    display(['finish converting class ' num2str(classi)]);
end