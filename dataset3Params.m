function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma.
%

C = 1;
sigma = 0.3;

a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
for i=1:8
    C = a(i);
    for j =1:8
        sigma = a(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if(i==1 && j==1)
            temp = error;
        else
            if(temp>error)
                temp = error;
                i1 = i;
                j1 = j;
            end
        end
    end
end
C = a(i1);
sigma = a(j1);

end
