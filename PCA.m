% Principal Component Analysis is one of the most widely used techniques for dimensionality re-
% duction. Dimensionality reduction in PCA is achieved by decomposing the eigen values of the
% variance covariance matrix. The decomposed eigen values give way to principal components. The
% total number of principal components would be the same as dimension of the given data, but
% then fewer principal components are selected such that the total number of principal components
% would be just enough to capture more than a certain percent percent of variance. This amount
% of variance to be selected is very arbitrary and may vary from case to case. In general the error
% increase from dimensionality reduction can be expressed as sum of the neglected components. So
% selecting the amount of variance captured is very subjective.

load cho_dataset
[pn, std_p] = mapstd(choInputs);
[tn, std_t] = mapstd(choTargets);

% I have investigated Cholesterol dataset which contains twenty-one spectral
% measurements of 264 blood samples and three kinds of cholesterol for each blood sample. PCA
% was conducted on the data set and components contributing less than 0.0001 of the total variance
% were discarded. So, out of the 21 components only the first 4 components were selected.

[pp, pca_p] = processpca(pn, 'maxfrac', 0.001);
[m, n] = size(pp)
%[coeff,score,latent] = pca(choInputs);
test_ind = 2:4:n;
val_ind = 4:4:n;
train_ind = [1:4:n 3:4:n];
net = fitnet(5);
net.trainFcn = 'trainlm';
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', train_ind, ...
    'valInd', val_ind, ...
    'testInd', test_ind);
[net, tr] = train(net, pn, tn);
Yhat_train = net(pn(:, train_ind));
Yhat_test = net(pn(:, test_ind));
MSE_test = perform(net,tn(:,test_ind),Yhat_test)
net_pca = configure(net, pp, tn);
net_pca.trainFcn = 'trainlm';
[net_pca, tr_pca] = train(net_pca, pp, tn);
Yhat_testpca = net_pca(pp(:, test_ind));
MSE_testpca = perform(net_pca,tn(:,test_ind),Yhat_testpca)

% the error increases slightly when the reduced data set is used.
% But that is the trade-off made so that a reduced input is used. One of the drawback of Bayesian
% Regularization is that it takes more time than Levenberg-Marquardt as it checks for the minimum
% weight with minimum residuals. Although Bayesian regularization took more time to converge,
% the difference between the time taken for the original input and the reduced input for Bayesian
% Regularization was significantly different. Hence, the findings corroborate the popularity in data
% reduction when time consuming optimization algorithms are used.


