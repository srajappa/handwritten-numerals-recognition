function y_final = test_lr(W_ans)
%% Logistic Regression Training Phase
load('test_data.mat');

M = 512;                            % Number of features
K = 10;                             % classes
N = size(test_data,1);                          % Total number of inputs
cntr = 0; 

%% Model the variables for further operations

% Output Variable
j = 1;
T = zeros(N,10);
for i = 1:10
    T(j:i*150,i) = 1;              % Perform sequential assignment per unit
    j = i * 150;                   % feature
end

% Create PHI values 
PHI = test_data;
PHI = [ones(N,1) PHI(:,1:512)];     % Bias Value

%% Perform the Error calculation

Y_ans = PHI*W_ans;

ind2sub(Y_ans,max(Y_ans,[],2));

[~,num]=max(Y_ans,[],2);
%Create the Output Prediction matrix.
expected_ans = zeros(1500,10);
for i=1:N
    j=num(i,1);
    expected_ans(i,j)=1;
end
%Calculate Error Rate and Reciprocal Rank.
misses = (sum(sum(abs(T-expected_ans))))/2;
error_rate= (misses/N).*100;

for i=1:N
    if T(i,:)==expected_ans(i,:)
        break;
    end
end
[~,T_label]=max(expected_ans,[],2);
T_label=T_label-1;
fprintf('Error Rate for logistic Regression is %d \n', error_rate );
fprintf('Reciprocal Ranks is %d \n', 1/i );
y_final=T_label;
