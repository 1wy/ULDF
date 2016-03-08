% Version 1.000
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

function [err] = mnistdisp(digits,PatchSize)
% display a group of MNIST images 
col=PatchSize;
row=PatchSize;

[dd,N] = size(digits);
imrow = dd/(col*row);
imdisp=zeros(imrow*row,N*col);



for ii = 1:imrow
   for jj = 1:N
        img1 = reshape(digits(1+(ii-1)*col*row:ii*col*row,jj),row,col);
        img2(((ii-1)*row+1):(ii*row),((jj-1)*col+1):(jj*col)) = img1';
   end
end


imagesc(img2,[0 1]); colormap gray; axis equal; axis off;
drawnow;
err=0; 

