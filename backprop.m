function backprop(para)

maxepoch = para.ftmaxepoch;
data_dir = para.data_dir;
PatchSize = para.PatchSize;
fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');


load([data_dir '/' 'mnistvh']);
load([data_dir '/' 'mnisthp']);
load([data_dir '/' 'mnisthp2']);
load([data_dir '/' 'mnistpo']);

makebatches;
[numcases numdims numbatches]=size(batchdata);
N=numcases;

%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w4=[hidtop; toprecbiases];
w5=[hidtop'; topgenbiases];
w6=[hidpen2'; hidgenbiases2];
w7=[hidpen'; hidgenbiases];
w8=[vishid'; visbiases];
% load([data_dir '/' 'mnist_weights']);
%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w5,1)-1;
l6=size(w6,1)-1;
l7=size(w7,1)-1;
l8=size(w8,1)-1;
l9=l1;
test_err=[];
train_err=[];


for epoch = 1:maxepoch
    
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    [numcases numdims numbatches]=size(batchdata);
    N=numcases;
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)];
        data = [data ones(N,1)];
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
        w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
        w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
        w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
        w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
        dataout = 1./(1 + exp(-w7probs*w8));
        err= err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 ));
    end
    train_err(epoch)=err/numbatches;
    
    %%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%% DISPLAY FIGURE TOP ROW REAL DATA BOTTOM ROW RECONSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf(1,'Displaying in figure 1: Top row - real data, Bottom row -- reconstructions \n');
    
    if epoch==1
        orioutput=[];
        for ii=1:15
            orioutput = [orioutput data(ii,1:end-1)'];
        end
        close all
        figure('Position',[100,600,1000,200]);
    else
        figure(1)
    end
    recoutput = [];
    for ii=1:15
        recoutput = [recoutput dataout(ii,:)'];
    end
    output = [orioutput;recoutput];
    mnistdisp(output,PatchSize);
    drawnow;
    fprintf(1,'Before epoch %d Train squared error: %6.3f \n',epoch,train_err(epoch));
    
    tt=0;
    for batch = 1:numbatches/10
        fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt=tt+1;
        data=[];
        for kk=1:10
            data=[data
                batchdata(:,:,(tt-1)*10+kk)];
        end
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=3;
        VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
        Dim = [l1; l2; l3; l4; l5; l6; l7; l8; l9];
        
        [X, fX] = minimize(VV,'CG_MNIST',max_iter,Dim,data);
        
        w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
        xxx = (l1+1)*l2;
        w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
        xxx = xxx+(l2+1)*l3;
        w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
        xxx = xxx+(l3+1)*l4;
        w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
        xxx = xxx+(l4+1)*l5;
        w5 = reshape(X(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
        xxx = xxx+(l5+1)*l6;
        w6 = reshape(X(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
        xxx = xxx+(l6+1)*l7;
        w7 = reshape(X(xxx+1:xxx+(l7+1)*l8),l7+1,l8);
        xxx = xxx+(l7+1)*l8;
        w8 = reshape(X(xxx+1:xxx+(l8+1)*l9),l8+1,l9);
        
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    save([data_dir '/' 'mnist_weights'], 'w1','w2','w3','w4','w5','w6','w7','w8');
    save([data_dir '/' 'mnist_error'],'test_err','train_err');
    fprintf('finish fine-tuning\n');
end



