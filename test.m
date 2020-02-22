load simpleDL.mat trainedNet lgraph
digitDatasetPath = fullfile('C:\Users\amr rashed\Desktop\Matlab\ADBTEST');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
img = readimage(imds,100);
actualLabel = imds.Labels(100);
predictedLabel = trainedNet.classify(img);
imshow(img);
title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])