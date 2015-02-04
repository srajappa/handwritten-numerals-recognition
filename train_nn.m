function [W1,W2,E_result] = train_nn(times)
%% Logistic Regression Training Phase
load('train_data.mat');

M = 512;                            % Number of features
H = (2./3).* 513;                    % Hidden Layers
K = 10;                             % classes
N = size(train_data,1);             % Total number of inputs
cntr = 0; 
num_steps = 0; 

n1 = 0.00001;
n2 = 0.0000001;
E_cross = inf;                      % The cross entropy value is initially INF
Error = inf;
E_result = [];

Y = zeros(N,K);

%% Model the variables for further operations

% Output Variable
j = 1;
T = zeros(N,10);

for i = 1:10
    T(j:i*2000,i) = 1;              % Perform sequential assignment per unit
    j = i * 2000;                   % feature
end

PHI = train_data;
PHI = [ones(size(PHI,1),1) PHI(:,1:512)];

W_l1 = rand(M+1,H)-0.5;
W_l2 = rand(H,K)-0.5;

W_n1 = W_l1;
W_n2 = W_l2;

W_u1 = zeros(M+1,H);
W_u2 = zeros(H,K);

while (num_steps <= times && ((Error > 0 )&& ~isnan(E_cross)))
    % Calculate for the hidden layer 1
    Aj = PHI*(W_l1);
    sigma_Aj = 1./(1+exp(-Aj));
    % Calculate for the hidden layer 2
    Ak = sigma_Aj*(W_l2);
    expAk = exp(Ak);
    sumAk = sum(expAk,2);
    
    %Using the above Calculate the output function 
    for i = 1:N
        for j = 1:K
            Y(i,j) = expAk(i,j)./sumAk(i,1);
        end
    end
    
    %Calculate the Error function (cross Entropy)
    
    E_cross = -sum(sum(T.*log(Y)));
    num_steps = num_steps + 1;
    E_result = [E_result;E_cross];
    fprintf('Cross_Entropy: %d Step %d\n',E_cross,num_steps);
    
    Error_k = Y-T;
    Error_j = (sigma_Aj.*(1-sigma_Aj)).*(Error_k*W_l2');

    delEk = PHI'*Error_j;
    delEj = sigma_Aj'*Error_k;
    
    % Performing the eta correction
    % Estimating the step size
    
    if (Error <= E_cross)
        cntr = 0; 
        n1 = n1+ 0.00001;
        n2 = n2+ 0.0000001;
    else
        cntr = cntr+1;
        if cntr > 3
            cntr = 0;
            n1 = n1 + 0.0001;
            n2 = n2 + 0.000001;
        end
        W_u1 = W_n1;
        W_u2 = W_n2;
        Error = E_cross;
    end
    
    W_n1 = W_l1-(n1.*delEk);
    W_n2 = W_l2-(n2.*delEj);
    
    W_l1 = W_n1;
    W_l2 = W_n2;
end
W1 = W_u1;
W2 = W_u2;
