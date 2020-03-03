%clc;clear
%https://www.mathworks.com/help/deeplearning/examples/transfer-learning-using-alexnet.html
%alex net,resnet,darknet importer,google net,cnn,tensorflow and keras models
%deepNetworkDesigner
digitDatasetPath = fullfile('C:\Users\amr rashed\Desktop\Matlab\ADBTEST');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
net = alexnet;
%analyzeNetwork(net)
%net.Layers
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
options = trainingOptions('rmsprop', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(imdsTrain,layers,options);
[YPred,scores] = classify(netTransfer,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)%0.9867
%accuracy = mean(YPred == imdsValidation.Labels)
%%save Network

%% Try to classify something else
img = readimage(imds,100);
actualLabel = imds.Labels(100);
predictedLabel = netTransfer.classify(img);
imshow(img);
title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])
