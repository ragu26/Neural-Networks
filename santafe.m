% The Santa Fe data set is obtained from a chaotic laser which can be described as a nonlinear
% dynamical system. 1000 data points were provided to train a neural network in a feed-forward
% mode. Based on the trained network 100 data points were generated and these were compared
% with subsequent 100 points which were given as a prediction set.
fid = fopen('E:\Mstat Courses\DMNN\Ex 2\lasertrain.dat','r');
[lasertr]= fscanf(fid, ['%f']);
fclose(fid);
fid = fopen('E:\Mstat Courses\DMNN\Ex 2\laserpred.dat','r');
[laserpr]= fscanf(fid, ['%f']);
fclose(fid);

%NN parameters
% As can be observed from the training data there is a pattern every 20 data-points consecutively
% till 200th data point and the same pattern iterates till 600th data point. Hence it will be logical
% to select a lag value of 20 so that the lag can capture any seasonality in the dataset.
lags=20;
wd=0.0000001;
logl=lasertr;
xi=tonndata(logl,false,false);
x_pred=tonndata(laserpr,false,false);

% In this time series prediction problem, a NARX model without exogenous input, thus a NAR
% (Non-linear AutoRegressive) model, was employed.
net = narnet(1:lags,75);
% The idea of lag in time series data is that the lag operation
% remembers the value being passed to it and returns as its result, the value passed to it on the
% previous call i.e. it works on a dierence of indices.

net.performParam.regularization = 0.00005;
[Xs,Xi,Ai,Ts] = preparets(net,{},{},xi);
net.divideFcn='divideind';
net.layers{1}.transferFcn = 'tansig';

% the  test and validation set was defined from staggered block such that for every 100 data points the
% first 75 represented the training set and rest 25 represented the validation set. With the latter
% method (staggered training and validation data), prediction performance improved considerably.
% The latter method might be a better way to include the training and validation set as it might
% capture the periodic trends which would be overlooked by lag. So I chose the latter method to
% check for the properties of the algorithms and prediction.
net.divideParam=struct('trainInd',[1:70,101:170,201:270,301:370,401:470,501:570,601:670,701:770,801:870,901:970],...
    'valInd',[76:90, 176:190,276:290,376:390,476:490,576:590,676:690,776:790,876:890,976:990],...
    'testInd',[91:100, 191:200,291:300,391:400,491:500,591:600,691:700,791:800,891:900,991:1000]);%notestset
net.trainFcn = 'trainlm';
net = configure(net,x,y);
net.b{1} = wd.*rand(75,1);
%       net.b{2} = wd(k).*randn(1,1);
net.iw{1} = wd.*rand(75,20);
net = closeloop(train(net,Xs,Ts,Xi,Ai));
prediction = nan(100+lags,1);
prediction = tonndata(prediction,false,false);
prediction(1:lags) = xi(end-(lags-1):end);
[xc,xic,aic,tc] = preparets(net,{},{},prediction);
prediction = fromnndata(net(xc,xic,aic),true,false,false);
plot(laserpr, 'g-');
hold on;
plot(prediction, 'r-');
legend({'Given Prediction','Estimated Prediction'})
hold off;

% Levenberg-Marquardt algorithm gave the best results as well as the fastest results. This is
% understandable as the algorithm works towards minimizing the residuals at each iteration.
% • Scaled Conjugate Gradient gave bad results for the right tail although it converged faster.
% The reason for faster convergence being the fact that it does not have the Hessian to be
% computed at every iteration.
% • BFGS Quasi-Newton took very long time to complete the training and had a poor fit.
% • Resilient Back-propagation gave the worst results out of all. The reason can be that the
% error terms were back-propagated and with each iteration the error gets carried forward.