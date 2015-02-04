function y_final = test_lr(W1,W2)
%% Logistic Regression Training Phase
load('test_data.mat');

M = 512;                            % Number of features
K = 10;                             % classes
N = size(test_data,1);                          % Total number of inputs
cntr = 0; 
answer = zeros(N,K);
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

Aj = PHI*(W1);
sigma_Aj = 1./(1+exp(-Aj));
fprintf('Size %d-%d W %d-%d\n',size(sigma_Aj,1),size(sigma_Aj,2),size(W2,1),size(W2,2));
Ak = sigma_Aj*(W2);


expAk = exp(Ak);
sumAk = sum(expAk,2);

expected_ans = zeros(1500,10);
for i=1:N
    for j=1:10;
        expected_ans(i,j) = expAk(i,j)./sumAk(i,1);
    end
end


[~,n] = max(expected_ans,[],2);

for i= 1:N
    answer(i,n(i,1)) = 1;
end

miss = (sum(sum(abs(T-answer))))/2;
error = (miss/size(PHI,1)).*100;

for i = 1:size(PHI,1)
    if T(i,:) == expected_ans(i,:)
        break;
    end
end

[~,T_label] = max(expected_ans,[],2);
T_label = T_label-1;

fprintf('the Error Rate for the Neural Network model is %d \n', error );
fprintf('the Reciprocal Rank for the Neural Network model is %d \n', 1/i );

y_final=T_label;

