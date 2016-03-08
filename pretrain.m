function pretrain(para)

maxepoch = para.ptmaxepoch;  
numhid = para.numhid;
numpen = para.numpen;
numpen2 = para.numpen2;
numopen = para.numopen;
data_dir = para.data_dir;

makebatches;
[numcases, numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbm;
hidrecbiases=hidbiases;
save([data_dir '/' 'mnistvh.mat'], 'vishid', 'hidrecbiases', 'visbiases');

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
batchdata=batchposhidprobs;
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save([data_dir '/' 'mnisthp.mat'], 'hidpen', 'penrecbiases', 'hidgenbiases');

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
batchdata=batchposhidprobs;
numhid=numpen2;
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save([data_dir '/' 'mnisthp2.mat'], 'hidpen2', 'penrecbiases2', 'hidgenbiases2');

fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numopen);
batchdata=batchposhidprobs;
numhid=numopen;
restart=1;
rbmhidlinear;
hidtop=vishid; toprecbiases=hidbiases; topgenbiases=visbiases;
save([data_dir '/' 'mnistpo.mat'], 'hidtop', 'toprecbiases', 'topgenbiases');
end