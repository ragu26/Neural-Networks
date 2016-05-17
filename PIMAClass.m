% The data set contains 768 records of female Pima Indians. Several MLP's were trained for this
% binary classification problem by taking class labels +1 and -1 as target values in a nonlinear
% regression.
myVars = {'Y','Xnorm'};
pid = load('pidstart.mat',myVars{:});

% the classifications cannot be separated by a linear separator i.e.
% a linear separator cannot define the boundaries for the two classes plotted in red and blue. The
% hidden layer transfer function and output layer transfer function are changed to a log sigmoid
% function, in order to produce output patterns consisting of 0's and 1's.

target = hardlim(pid.Y)';
x=pid.Xnorm';
lsize=2;
net=patternnet(lsize);
net.performParam.regularzation = 0.000001;

% The dataset was divided into training data comprising 60% of the data, validation data comprising 20% of the data and
% test data comprising 20% of the data. The data were divided into training, test and validation
% randomly.
net.divideParam.trainRatio = 60 / 100 ;
net.divideParam.valRatio = 20 / 100 ;
net.divideParam.testRatio = 20 / 100 ;
net.layers{1}.transferFcn = 'logsig';
net.outputs{1}.transferFcn = 'logsig';
net.TrainFcn='trainbr';
[net, tr]=train(net, x, target);
output=net(x);
error=gsubtract(target,output);
performance=perform(net,target,output);

% Bayesian Regularization gave the best results for neurons 2
% and 5, where as Levenberg-Marquardt gave the best results for 10 neurons. All other algorithms
% classified the data poorly. The performances of Levenberg-Marquardt and Bayesian Regular-
% ization are similar at some point(5 neurons) because Bayesian Regularization is an extension of
% Levenberg-Marquardt. Although Levenberg-Marquardt algorithm minimizes the residual factors,
% it is less prone to detect outliers. Not just the outliers, Levenberg-Marquardt algorithm is very
% sensitive to initial weights which makes Bayesian Regularization better. Bayesian Regularization
% expands the cost function to search not only for the minimal error, but also for the minimal
% error using the minimal weights. This in turn avoids cross validation. It has to be noted that
% for 10 neurons Levenberg-Marquardt performs better than Bayesian Regularization. This can
% be caused by over-generalization in which too many neurons provide a greater
% exibility which always is not beneficial.
