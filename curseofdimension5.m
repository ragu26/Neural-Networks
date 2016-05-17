
% To test the non linear mapping with five dimensional input, 3 sets of input data were generated
% viz. training data, validation data and test data. The training set comprised of 10 data points
% between the values -5 to 5 including the boundary values. The validation set comprised of 4
% data points between between the values -4.5 to 4.5 including the boundary values. The test set
% comprised of 4 data points between between the values -4.75 to 4.75 including the boundary
% values. Although training data, validation data and test data have close boundary values they do
% not share any data points.The algorithms beyond 250 neurons took a very long time to complete
% and none f the algorithms achieved convergence.

%Data Generation
d1=linspace(-5,5,20);
d2=linspace(-5,5,20);
d3=linspace(-5,5,20);
d4=linspace(-5,5,20);
d5=linspace(-5,5,20);
[gen1,gen2,gen3,gen4,gen5]=ndgrid(d1,d2,d3,d4,d5);

%training set
tr_input=([gen1(:), gen2(:), gen3(:), gen4(:), gen5(:) ].');
intermed=tr_input.*tr_input;
tr_output=sinc(sqrt((intermed(1,:)+intermed(2,:)+intermed(3,:)+intermed(4,:)+intermed(5,:))));

%validation set
val_d1=linspace(-4.5,4.5,6);
val_d2=linspace(-4.5,4.5,6);
val_d3=linspace(-4.5,4.5,6);
val_d4=linspace(-4.5,4.5,6);
val_d5=linspace(-4.5,4.5,6);
[val_gen1,val_gen2,val_gen3,val_gen4,val_gen5]=ndgrid(val_d1,val_d2,val_d3,val_d4,val_d5);
val_input=([val_gen1(:), val_gen2(:), val_gen3(:), val_gen4(:), val_gen5(:) ].')
val_intermed=val_input.*val_input;
val_output=sinc(sqrt((val_intermed(1,:)+val_intermed(2,:)+val_intermed(3,:)+val_intermed(4,:)+val_intermed(5,:))));
input=[tr_input val_input];
output=[tr_output val_output];

% test set
test_d1=linspace(-4.75,4.75,4);
test_d2=linspace(-4.75,4.75,4);
test_d3=linspace(-4.75,4.75,4);
test_d4=linspace(-4.75,4.75,4);
test_d5=linspace(-4.75,4.75,4);
[test_gen1,test_gen2,test_gen3,test_gen4,test_gen5]=ndgrid(test_d1,test_d2,test_d3,test_d4,test_d5);

test_input=([test_gen1(:), test_gen2(:), test_gen3(:) , test_gen4(:) , test_gen5(:)  ].')
test_intermed=test_input.*test_input;
test_output=sinc(sqrt((test_intermed(1,:)+test_intermed(2,:)+test_intermed(3,:)+test_intermed(4,:)+test_intermed(5,:))));

%NN parameters
net=fitnet(200,'trainscg');
net.divideFcn='dividerand';
% net.divideParam=struct('trainInd',[1:2500],...
%          'valInd',[2501:2900],...
% % %         'testInd',[]);%notestset
net.performParam.regularization = 0.000001 ;
% [net,tr]=train(net,input,output);
%
test_yhat = net(test_input);
%
plot3(test_input(1,:),test_input(2,:),test_output,'b*');
title('Five Dimensional Sinc Function - Dimension One and Two')
xlabel('Input - First Dimension')
ylabel('Input - Second Dimension')
zlabel('Output')
hold on;
plot3(test_input(1,:),test_input(2,:),test_yhat,'ro');
legend({'Generated Output','Estimated Output'});
hold off;

figure
plot3(test_input(3,:),test_input(4,:),test_output,'b*');
title('Five Dimensional Sinc Function - Dimension Three and Four')
xlabel('Input - Third Dimension')
ylabel('Input - Fourth Dimension')
zlabel('Output')
hold on;
plot3(test_input(3,:),test_input(4,:),test_yhat,'ro');
legend({'Generated Output','Estimated Output'});
hold off;


figure
plot(test_input(5,:),test_output,'b*');
title('Five Dimensional Sinc Function - Dimension Five')
xlabel('Input - Fifth Dimension')
ylabel('Output')
hold on;
plot(test_input(5,:),test_yhat,'ro');
legend({'Generated Output','Estimated Output'});
hold off;

% When executed for Levenberg-Marquardt, the
% algorithm took a long time to start. This is understandable as the algorithm check for minimizing
% the residual, it is computationally intensive. I tried with Scaled Conjugate Gradient algorithm
% and the algorithm converged although the quality of prediction was very poor.
% The reason for Scaled Conjugate algorithm converging is that it does not have Hessian Matrix to
% be iterated every time and so it computes the iteration faster. Although it converges the quality
% of prediction turns out to be extremely poor that the convergence does not make sense.