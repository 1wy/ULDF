digitdata=[]; 
targets=[]; %label,its dimention is numbel of classes

for classi = 0:9
    load([data_dir '/digit' num2str(classi)]);
    digitdata = [digitdata; D];
    classitar = zeros(1,10);
    classitar(classi+1) = 1;
    targets = [targets; repmat(classitar, size(D,1), 1)];
end
digitdata = digitdata/255;  %normalize, origin data range between 0 to 255

totnum=size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %set seed, so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdata,2);
batchsize = 100;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 10, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets;

%%% Reset random seeds 
% rand('state',sum(100*clock));  %uniform distribution
% randn('state',sum(100*clock)); %normal distribution
