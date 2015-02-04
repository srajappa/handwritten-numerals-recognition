function [W_ans,E_result] = train_lr(times)
%% Logistic Regression Training Phase
load('train_data.mat');

M = 512;                            % Number of features
K = 10;                             % classes
N = 20000;                          % Total number of inputs
cntr = 0; 

neeta = 0.00001;                    % Decides the step size
E_cross = inf;                      % The cross entropy value is initially INF
Error = inf;

%% Model the variables for further operations

% Output Variable
j = 1;
T = zeros(N,10);

for i = 1:10
    T(j:i*2000,i) = 1;              % Perform sequential assignment per unit
    j = i * 2000;                   % feature
end

% Arbitrary value of W
W = rand(M+1,K);                    % Assign random values
W(1,:) = 1;                         % Bias Value

% Create PHI values 
PHI = train_data;
PHI = [ones(N,1) PHI(:,1:512)];     % Bias Value

%% Perform the Error calculation
num_steps = 0; 
W_next = zeros(M+1,K);
W_update = zeros(M+1,K);
W_ans = zeros(M+1,K);

E_result = [];


while(num_steps <= times && (Error >= 2 )&& ~isnan(E_cross))
    % Creating Analysis function 
    a = PHI*W;
    exp_a = exp(a);
    exp_sum = sum(exp_a,2);
    %Create Ynk
    Y = ones(1,K);
    for i = 1:N
        for j = 1:K
            Y(i,j) = exp_a(i,j)./exp_sum(i,1);
        end
    end
    
    %Error Del
    del = (PHI'*(Y-T));
    
    %Performing Gradient Descent
    W_next = W - (neeta.* del);
    num_steps = num_steps+1;
    
    %Calculate the cross entropy
    E_cross = -1 * sum(sum(T.*log(Y)));
    fprintf('Error: %d Step num: %d\n',E_cross,num_steps);
    E_result = [E_result;E_cross];
    
    if(Error <= E_cross)
        cntr = 0; 
        neeta = 0.00001;
    else
        cntr = cntr +1;
        if cntr > 3
            cntr = 0;
            neeta = neeta + 0.0001;     %Increment the step size
        end
        W_update = W_next;
        Error = E_cross;
    end
    W = W_next;
end
figure;
xlabel('Iterations');
ylabel('Cross Entropy');
plot(1:times+1,E_result,'r');
title('Cross-Entropy (Error) vs Total number of Iterations');
W_ans = W_update;
