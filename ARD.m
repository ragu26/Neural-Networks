% ARD is a technique that detects the relevant components of the input, It gives an option to
% discard the non-relevant portion of the input. In the previous section in the PIMA Indians
% classification, it was shown that the Bayesian Regularization was superior to Levenberg Mar-
% quardt algorithm and a complexity governing function gamma
%  was introduced. The cost function of
% the Bayesian regularization contained one of the hyperparameters alpha. In ARD, instead of having
% one hyperparameter alpha for the whole MLP network, one can consider separate hyperparameters
% alpha_i for the set of weights associated with each input node of the network. After training the
% different alpha_i values are obtained. From these values we can investigate the relative importance
% of the inputs with respect to each other. The individual alpha_i values correspond to the inverse of

ion = load('ionstart.mat');
inputs = ion.Xnorm;
targets = hardlim(ion.Y);
nin = 33;
nhidden = 5;
nout = 1;
aw1 = 0.01*ones(1,nin);
ab1 = 0.01;
aw2 = 0.01;
ab2 = 0.01;
beta = 50.0;
train_ind=[1:6:351,2:6:351,3:6:351,4:6:351];
test_ind=[5:6:351,6:6:351];
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'logistic', prior);

nouter = 2;
ninner = 10;
options = zeros(1,18);
options(1) = 1;
options(2) = 1e-7;
options(3) = 1e-7;
options(14) = 300;

for k = 1:nouter;
    net = netopt(net, options, inputs(train_ind,:), targets(train_ind,:), 'scg');
    [net, gamma] = evidence(net, inputs(train_ind,:), targets(train_ind,:), ninner);
end;

[outputs, z] = mlpfwd(net, inputs(test_ind,:));

figure, plotconfusion(targets(test_ind,:)',outputs');
title('Original Input')
figure, plotroc(targets(test_ind,:)',outputs');
title('Original Input')
[X,Yc,T,AUCo] = perfcurve(targets(test_ind,:),outputs,1);
v = [outputs targets(test_ind,:)];

%%Reduced Input training
ind = 1:33;
w = net.alpha(1:33);
w = [ind' w];
w = sortrows(w,2);
ind = w(1:6,1);
inputs2 = ion.Xnorm(:,ind);
ninr = 6;
nhiddenr = nhidden;
noutr = nout;
aw1r = 0.01*ones(1,ninr);
ab1r = 0.01;
aw2r = 0.01;
ab2r = 0.01;
betar = 50.0;

priorr = mlpprior(ninr, nhiddenr, noutr, aw1r, ab1r, aw2r, ab2r);
netr = mlp(ninr, nhiddenr, noutr, 'logistic', priorr);
nouterr = 2;
ninnerr = 10;
options = zeros(1,18);
options(1) = 1;
options(2) = 1e-7;
options(3) = 1e-7;
options(14) = 300;


for k = 1:nouterr;
    netr = netopt(netr, options, inputs2(train_ind,:), targets(train_ind,:), 'scg');
    [netr, gammar] = evidence(netr, inputs2(train_ind,:), targets(train_ind,:), ninnerr);
end;
[outputs2, zr] = mlpfwd(netr, inputs2(test_ind,:));
figure, plotconfusion(targets(test_ind,:)',outputs2');
title('Reduced Input')
figure, plotroc(targets(test_ind,:)',outputs2');
title('Reduced Input')
[X,Yc,T,AUCr] = perfcurve(targets(test_ind,:),outputs2,1);
vr = [outputs2 targets(test_ind,:)];

