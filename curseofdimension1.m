% As the number of descriptive parameters about the information to be trained increases, the di-
% mension also increases. With the addition of each feature, a parameter gets added to the input
% data otherwise called dimension. The fundamental concept of neural networks is to map infor-
% mation from one space to another. In order to perform the mapping all the inputs pertaining to
% the input space is required, in other words more the training information better the training and
% subsequent predictions,yet this advantage comes with a big drawback of dimension increase. As
% the dimension increases, more observations are required to train the network which otherwise is
% called Curse of Dimensionality.

% To test the non linear mapping with one dimensional input, 3 sets of input data were generated
% viz. training data, validation data and test data. The training set comprised of 100 data points
% between the values -5 to 5 including the boundary values. The validation set comprised of 40
% data points between the values -4.5 to 4.5 including the boundary values. The test set comprised
% of 40 data points between the values -4.75 to 4.75 including the boundary values. Although
% training data, validation data and test data have close boundary values they do not share any
% data points.

%Data Generation
d1=linspace(-5,5,80);
d2=linspace(-5,5,80);
[gen1,gen2]=meshgrid(d1,d2);
%Z=sinc(sqrt(X.*X+Y.*Y));
tr_input=([gen1(:), gen2(:) ].')
intermed=tr_input.*tr_input;
tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:))));

val_d1=linspace(-4.5,4.5,40);
val_d2=linspace(-4.5,4.5,40);
[val_gen1,val_gen2]=meshgrid(val_d1,val_d2);
val_input=([val_gen1(:), val_gen2(:) ].')
val_intermed=val_input.*val_input;
val_output=sinc(sqrt((val_intermed(1,:)+val_intermed(2,:))));
input=[tr_input val_input];
output=[tr_output val_output];

test_d1=linspace(-4.75,4.75,40);
test_d2=linspace(-4.75,4.75,40);
[test_gen1,test_gen2]=meshgrid(test_d1,test_d2);
%Z=sinc(sqrt(X.*X+Y.*Y));
test_input=([test_gen1(:), test_gen2(:) ].')
test_intermed=test_input.*test_input;
test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:))));

net=fitnet(50,'trainlm');
% net.divideFcn='divideind';
% net.divideParam=struct('trainInd',[1:2500],...
%         'valInd',[2501:2900],...
%         'testInd',[]);%notestset
net.performParam.regularization = 0.000001 ;
[net,tr]=train(net,input,output);

test_yhat = net(test_input);
% Levenberg-Marquardt and quasi-Newton algorithms gave
% a near perfect fit while the Scaled Conjugate Gradient performed slightly worse than the two.

plot3(test_input(1,:),test_input(2,:),test_output,'b*');
title('Mexican Hat Function')
xlabel('Input - First Dimension')
ylabel('Input - Second Dimension')
zlabel('Output')
hold on;
plot3(test_input(1,:),test_input(2,:),test_yhat,'ro');
legend({'Generated Output','Estimated Output'});
hold off;
display(tr.best_perf)

% Resilient back-propagation came out to be the worst fit out of all four algorithms. There weren't
% any noticeable time differences among the algorithms although Levenberg-Marquardt and scaled
% conjugate gradient converged faster than the rest.