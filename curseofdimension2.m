% TO test the non linear mapping with two dimensional input, 3 sets of input data were generated
% viz. training data, validation data and test data. The training set comprised of 80 data points
% between the values -5 to 5 including the boundary values. The validation set comprised of 40
% data points between between the values -4.5 to 4.5 including the boundary values. The test set
% comprised of 40 data points between between the values -4.75 to 4.75 including the boundary
% values. Although training data, validation data and test data have close boundary values they
% do not share any data points.In a general trial and error method 50 neurons were found adequate
% to fit the two dimensional case.

%Data Generation
d1=linspace(-5,5,100);
tr_output=sinc(sqrt(d1.*d1));

%validation set
val_d1=linspace(-4.5,4.5,40);
val_output=sinc(sqrt(val_d1.*val_d1));
input=[d1, val_d1];
output=[tr_output, val_output];

%test set
test_d1=linspace(-4.75,4.75,40);
test_output=sinc(sqrt(test_d1.*test_d1));

%NN parameters
net=fitnet(10,'trainrp');
net.performParam.regularization = 0.000001 ;
net.divideFcn='divideind';
net.divideParam=struct('trainInd',[1:100],...
    'valInd',[101:140],...
    'testInd',[]);%notestset
[net,tr]=train(net,input,output);

test_yhat = net(test_d1);

plot(test_d1,test_output,'-b*');
title('One Dimensional sinc function');
xlabel('Input');
ylabel('Output');
hold on;
plot(test_d1,test_yhat,':ro');
xlabel('Input');
ylabel('Output');
legend({'Generated Output','Estimated Output'});
hold off;

% Levenberg-Marquardt performed the best when it came to prediction with test data.
% This information is corroborated by MSE for which Levenberg-Marquardt
% had the least value. Scaled Conjugate gradient had the worst performance as the estimated func-
% tion did not capture the tip of the hat and this fact is also corroborated by the highest value of
% MSE out of all four algorithms used.