RFdaily = 0;
% open RF manually
RF_full = table2array(RFdaily);

%% CDAX
R_CDAX_ae = csvread("yearly_portfolio_returns.csv",1,1); % Left : threshold, Right: diag
R_threshold_CDAX = R_CDAX_ae(:,1);
R_diag_CDAX = R_CDAX_ae(:,2);

R_CDAX_bm = csvread("yearly_portfolio_returns_oneoverN.csv",1,1); % Left : original, Right: 1/N
R_original_CDAX = R_CDAX_bm(:,1);
R_oneoverN_CDAX = R_CDAX_bm(:,2);

%% CAC
R_CAC_ae = csvread("yearly_portfolio_returns_CAC.csv",1,1); % Left : threshold, Right: diag
R_threshold_CAC = R_CAC_ae(:,1);
R_diag_CAC = R_CAC_ae(:,2);

R_CAC_bm = csvread("yearly_portfolio_returns_oneoverN_CAC.csv",1,1); % Left : original, Right: 1/N
R_original_CAC = R_CAC_bm(:,1);
R_oneoverN_CAC = R_CAC_bm(:,2);


% Select which portfolios to compare
R = [R_diag_CAC R_threshold_CAC];

% Remove zero rows
R( ~any(R,2), : ) = [];

% Align RF
[n_r, q1] = size(R);
[n_RF, q2] = size(RF_full);
RF = RF_full(n_RF-n_r+1:n_RF);
R = R - RF;

% Compute stats
[SR1, SR2] = sharpeRatio(R);
CR1 = sum(R(:,1));
CR2 = sum(R(:,2));
std1 = std(R(:,1));
std2 = std(R(:,2));

% Bootstrap
[pValue,DeltaHat,d,b,dStars,se] = bootInference(R, 100, 100000);
dStars = sort(dStars);
z = dStars(100000*0.95,1);
LB = DeltaHat - z*se;
UB = DeltaHat + z*se;




