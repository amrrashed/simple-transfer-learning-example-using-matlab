clc;clear
digitDatasetPath = fullfile('C:\Users\amr rashed\Desktop\Matlab\ADBTEST');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%outputSize = [224 224 3];
%auimds = augmentedImageDatastore(outputSize,imds);
%img2=zeros(224,224,3);
for i=1:length(imds.Labels)
img=readimage(imds,i);
 %imshow(img)
img1 = imresize(img,[227 227]);
%img2(:,:,1)=img1;
%img2(:,:,2)=img1;
%img2(:,:,3)=img1;
 imwrite(img1,cell2mat(imds.Files(i)))
end

