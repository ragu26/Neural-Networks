
% A training set of a sinusoidal signal (data scaled between -1 and +1 on each axis) was taken
% up for investigation in this subsection. Various factors that in
% uence training a neural network
% such as noise,data size, training algorithms, regularization parameters were investigated in detail.
% In particular, every instance corresponding to training performances of four training algorithms
% viz. back-propagation, conjugate gradient, quasi-Newton and Levenberg-Marquardt along with
% various values of noise and data size were compared with chosen values of other factors. The
% standard settings used were data size of 50, 200 or 1000 points and Standard Deviation of the
% Gaussian Noise taking a value 0.1,0.5 or 1 for each of the four algorithms compared. Mean Square
% Error was chosen as the performance measure to be tabulated although other measures were also
% noted during the training.

%%Create a training and validation set
noise=[0.1; 0.5; 1];
datasize=[50 ;200 ;1000];


for i=1:3
    for j=1:3
        %training set
        train_x=linspace(-1,1,datasize(j));
        train_y=sin(2*pi*train_x)+noise(i)*randn(size(train_x));
        
        %validation set
        val_x=linspace(-0.9,0.9,100);
        val_y=sin(2*pi*val_x)+0.5*randn(size(val_x));
        x=[train_x val_x];
        y=[train_y val_y];
        
        %%NN Parameters
        %%Create a network with 5 hidden neurons
        net=fitnet(5,'trainlm');
        net.performParam.regularization = 0 ;
        net.divideFcn='divideind';
        net.divideParam=struct('trainInd',1:100,...
            'valInd',101:200,...
            'testInd',[]);%notestset
        [net,tr]=train(net,x,y);
        
        %%Get approximated function on training set
        train_yhat=net(train_x);
        figure();
        
        %Trained function
        plot(train_x, train_y,'r*');
        hold on;
        
        %Approximated Function
        plot(train_x, train_yhat,'b-');
        hold on;
        
        
        %True Function
        plot(train_x, sin(2*pi*train_x),'g-');
        
        %Plot details
        title(strcat('For Noise',' ',num2str(noise(i)),' ',' and size of the data ',num2str(datasize(j)),'trainlm') );
        hold off;
        legend('Training Set','Approximated Function','True Function');
    end
end

%scg
for i=1:3
    for j=1:3
        train_x=linspace(-1,1,datasize(j));
        train_y=sin(2*pi*train_x)+noise(i)*randn(size(train_x));
        val_x=linspace(-0.9,0.9,100);
        val_y=sin(2*pi*val_x)+0.5*randn(size(val_x));
        x=[train_x val_x];
        y=[train_y val_y];
        
        %%Create a network with 5 hidden neurons
        net=fitnet(5,'trainscg');
        net.performParam.regularization = 0 ;
        net.divideFcn='divideind';
        net.divideParam=struct('trainInd',1:100,...
            'valInd',101:200,...
            'testInd',[]);%notestset
        [net,tr]=train(net,x,y);
        
        %%Get approximated function on training set
        train_yhat=net(train_x);
        figure();
        plot(train_x, train_y,'r*');
        hold on;
        
        plot(train_x, train_yhat,'b-');
        hold on;
        
        plot(train_x, sin(2*pi*train_x),'g-');
        title(strcat('For Noise',' ',num2str(noise(i)),' ',' and size of the data ',num2str(datasize(j)),'trainscg') );
        hold off;
        legend('Training Set','Approximated Function','True Function');
    end
end


%trainbfgs
for i=1:3
    for j=1:3
        train_x=linspace(-1,1,datasize(j));
        train_y=sin(2*pi*train_x)+noise(i)*randn(size(train_x));
        val_x=linspace(-0.9,0.9,100);
        val_y=sin(2*pi*val_x)+0.5*randn(size(val_x));
        x=[train_x val_x];
        y=[train_y val_y];
        
        %%Create a network with 5 hidden neurons
        net=fitnet(5,'trainbfg');
        net.performParam.regularization = 0 ;
        net.divideFcn='divideind';
        net.divideParam=struct('trainInd',1:100,...
            'valInd',101:200,...
            'testInd',[]);%notestset
        [net,tr]=train(net,x,y);
        
        %%Get approximated function on training set
        train_yhat=net(train_x);
        figure();
        plot(train_x, train_y,'r*');
        hold on;
        
        plot(train_x, train_yhat,'b-');
        hold on;
        
        plot(train_x, sin(2*pi*train_x),'g-');
        title(strcat('For Noise',' ',num2str(noise(i)),' ',' and size of the data ',num2str(datasize(j)),'trainbfg') );
        hold off;
        legend('Training Set','Approximated Function','True Function');
    end
end