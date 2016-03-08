% Version 1.000
%

clear all; close all; clc;

addpath('./Utils');
addpath('./liblinear-1.7-single/matlab');
%=============================Parameters =================================
para.ptmaxepoch = 30; %maxepoch for pretraining
para.ftmaxepoch = 10; %maxepoch for finetuning
%[numhid numpen numpen2 numopen] is the number of units in each layer of
%autoencoder
para.numhid = 150;                      
para.numpen = 75;
para.numpen2 = 30;
para.numopen = 15;
para.PatchSize = 10;   %size of patch
para.StepSize = 1;     %size of step when sampling patches
para.pyramid = [2];    %spatial pyramid
% para.pyramid = [1 2 4];
para.numcode = 100;
para.ImgSize = [28,28];
para.data_dir = ['patchdata' num2str(para.PatchSize)];
save para10 para;
%%=========================initialize with RBM=============================
fprintf(1,'Converting Raw files into Matlab format \n');
% converter;     %Converting Raw files into Matlab format
% converttopatch(para); %collect patches from training data
fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'Using %3i maxepoches for initialization\n', para.ptmaxepoch);
% pretrain(para);

%%====================finetune a deep autoencoder========================
fprintf(1,'finetuning a deep autoencoder\n');
fprintf(1,'Using %3i maxepoches for finetuning\n', para.ftmaxepoch);
load para10;
% backprop(para);
%  % =====================train and test model=============================
para.pyramid = [2];
models = train_model(para);
% save models models;
% load models;
[acc, proestimatetest] = test(models,para);
