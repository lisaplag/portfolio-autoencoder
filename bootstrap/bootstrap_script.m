RFdaily = 0;
% open RF manually
RF_full = table2array(RFdaily);

%% CDAX
R_CDAX_ae = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns.csv",1,1); % Left : threshold, Right: diag
R_threshold_CDAX = R_CDAX_ae(:,1);
R_diag_CDAX = R_CDAX_ae(:,2);

R_CDAX_bm = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns_oneoverN_CDAX.csv",1,1); % Left : original, Right: 1/N
R_original_CDAX = R_CDAX_bm(:,1);
R_oneoverN_CDAX = R_CDAX_bm(:,2);

R_CDAX_bm_mw = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns_oneoverN_maxweights_CDAX.csv",1,1); % Left : original, Right: 1/N
R_original_CDAX_mw = R_CDAX_bm_mw(:,1);
R_oneoverN_CDAX_mw = R_CDAX_bm_mw(:,2);

R_CDAX_final = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns_CDAX_mp.csv",1,1); % Left : original, Right: 1/N
R_diag_CDAX = R_CDAX_final(:,1);
R_original_mean_CDAX = R_CDAX_final(:,2);
R_threshold_CDAX = R_CDAX_final(:,3);

%% CAC
R_CAC_ae = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns_CAC.csv",1,1); % Left : threshold, Right: diag
R_threshold_CAC = R_CAC_ae(:,1);
R_diag_CAC = R_CAC_ae(:,2);

R_CAC_bm = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns_oneoverN_CAC.csv",1,1); % Left : original, Right: 1/N
R_original_CAC = R_CAC_bm(:,1);
R_oneoverN_CAC = R_CAC_bm(:,2);

R_CAC_bm_mw = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns_oneoverN_maxweights_CAC.csv",1,1); % Left : original, Right: 1/N
R_original_CAC_mw = R_CAC_bm_mw(:,1);
R_oneoverN_CAC_mw = R_CAC_bm_mw(:,2);

R_CAC_om = csvread("../data/results/Yearly_portfolio/yearly_portfolio_returns_om_CAC.csv",1,1); % Left : original, Right: 1/N
R_om_CAC = R_CAC_om(:,2);


% Select which portfolios to compare
R1 = R_threshold_CDAX;
R2 = R_original_mean_CDAX;

% Merge
s1 = size(R1);
s2 = size(R2);
if s1(1) > s2(1)
    R1 = R1(s1(1)-s2(1)+1:end);
end
if s1(1) < s2(1)
    R2 = R2(s2(1)-s1(1)+1:end);
end
    
R = [R1 R2];

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




