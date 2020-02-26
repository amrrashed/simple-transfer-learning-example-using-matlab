%clc;clear
%alex net,resnet,darknet importer,google net,cnn,tensorflow and keras models
%deepNetworkDesigner
digitDatasetPath = fullfile('C:\Users\amr rashed\Desktop\Matlab\ADBTEST');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
net = googlenet;
%analyzeNetwork(net)
%net.layers
numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'loss3-classifier',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);
%inputSize = net.Layers(1).InputSize;
% inputSize= [224 224];
% augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain);
% augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);
%learning functions sgdm,rmsprop,adam
options = trainingOptions('adam', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',8, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');
trainedNet = trainNetwork(imdsTrain,lgraph,options);

[YPred,probs] = classify(trainedNet,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)%0.9867
%accuracy = mean(YPred == imdsValidation.Labels)
%%save Network
save simpleDL.mat trainedNet lgraph
%% Try to classify something else
img = readimage(imds,100);
actualLabel = imds.Labels(100);
predictedLabel = trainedNet.classify(img);
imshow(img);
title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])
