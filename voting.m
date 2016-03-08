
addpath('./liblinear-1.7-single/matlab');
load labels;
load labelstra;
mulPatchSize = [9 10];
averprotra = zeros(60000,10);
averprotes = zeros(10000,10);
protrafea = [];
protesfea = [];
for i = 1:length(mulPatchSize)
    PatchSize = mulPatchSize(i);
    load(['patchdata' num2str(PatchSize) '/proestimatetest']); 
    load(['patchdata' num2str(PatchSize) '/proestimattra']); 
    averprotra = averprotra + proestimattra;
    averprotes = averprotes + proestimatetest;
    protrafea = [protrafea proestimattra];
    protesfea = [protesfea proestimatetest];
    [r,p]=max(proestimatetest');
    p=p-1;
    acc=sum(p==labels');
    display(100-acc/100);
end
% 
for j = 1:60000
    protrafea(j,:) = protrafea(j,:)/norm(protrafea(j,:));
end
    
for j = 1:10000
    protesfea(j,:) = protesfea(j,:)/norm(protesfea(j,:));
end
% averprotes = proestimate9+proestimate11+proestimate28 +proestimate8;
[r,p]=max(averprotes');
p=p-1;
voteacc=sum(p==labels');
display(['sum acc = ' num2str(100-voteacc)]);

   
votemodels = train(single(labelstra), single(protrafea), '-s 1 -c 1 -q');
[prelabel,tracc] = predict(labelstra,sparse(protrafea), votemodels);
display(['train acc = ' num2str(tracc)]);
[prelabel,voteacc] = predict(labels,sparse(protesfea), votemodels);
display(['vote acc = ' num2str(100-voteacc)]);

