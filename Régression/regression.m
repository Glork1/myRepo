clear all;
close all;
clc;

% DATES : 01/01/2015 ----- 31/12/2015 (1 an)
% Action Total (ou EDF) expliquée par :
% (100 jours de cotations par exemple = 100 obs yi)
% Rq : avec les cours actuels, large spectre d'obeservations
% Suivre démarche pdf régression

% ATTENTION: comparer ques les jours de cotations coincident (trous le week
% end)

% 1/ cours du cac40 (https://fr.finance.yahoo.com/q/hp?s=^FCHI)
% 2/ cours du dow jones https://research.stlouisfed.org/fred2/series/DJIA/downloaddata
% 3/ cours d'Airfrance (qd pétrole augmente, moins de bénéf (https://fr.finance.yahoo.com/q/hp?s=AF.PA)
% 4/ cours d'une entreprise concurrente - BP (https://fr.finance.yahoo.com/q/hp?s=BP.L)
% 5/ cours d'entreprises qui n'ont (vraiment) rien à voir, par exemple  DANONE (https://fr.finance.yahoo.com/q/hp?s=BN.PA)
% 6/ cours d'entreprises qui n'ont (apparemment) rien à voir, par exemple  L'OREAL (https://fr.finance.yahoo.com/q/hp?s=OR.PA&b=1&a=00&c=2015&e=31&d=11&f=2015&g=d)
% 7/ cours de Michelin (pneus<- pétrole) (https://fr.finance.yahoo.com/q/hp?s=OR.PA)
% 8 cours d'Orange https://fr.finance.yahoo.com/q/hp?s=ORA.PA&b=1&a=00&c=2015&e=31&d=11&f=2015&g=d
% 9/cours Vinci https://fr.finance.yahoo.com/q/hp?s=DG.PA&b=1&a=00&c=2015&e=31&d=11&f=2015&g=d
% 10/prix du baril de brut Europe (taper : https://research.stlouisfed.org/fred2/series/DCOILBRENTEU/downloaddata)


% de 01/01/2010 à 01/01/2015, chaque jour


%Régression linéaire simple
% Variable observée (cours d'ouverture de total (2eme colonne))
load total.csv

% Variables explicatives
load cac40.csv; %1
load dowjones.csv; %2
load af.csv; %3
load bp.csv; %4
load danone.csv; %5
load loreal.csv; %6
load michelin.csv; %7
load orange.csv; %8
load vinci.csv; %9
load brut.csv; %10

n=size(total(:,2));
n=n(1)
p = 10; % nombre de régresseurs

% Estimateurs MC
phi = [cac40(:,2),dowjones(:,2),af(:,2),bp(:,2),danone(:,2),loreal(:,2),michelin(:,2),orange(:,2),vinci(:,2),brut(:,2)];
theta_chapeau = pinv((phi'*phi))*phi'*total(:,2)
Ychapeau= phi*theta_chapeau;

% Histogramme des résidus
Y = total(:,2);
e= Y-Ychapeau;
hist(e);

% Intervalles de confiance à 95% sur theta_j
e = Y-Ychapeau;
MSE = e'*e/(n-p); %non n-p+1
sigma_hat = sqrt(MSE);
alpha = 0.05;
quantile = -qt(alpha/2,n-p); %qt: quantile de la loi de Student à n-(p+1) degrés de liberté
intervalle = zeros(p,2);
M = pinv(phi'*phi);
for j=1:p
    a = quantile*sigma_hat*sqrt(M(j,j));
    intervalle(j,:) = theta_chapeau(j)+[-a,a];
end
intervalle

% Sélection de variables
% p-valeurs
std = sqrt(diag(MSE*inv(phi'*phi)));
t = theta_chapeau./std % fournit un test de student pour chaque coefficient
p_valeur = 1-tcdf(t,n-p)
% Plus la p-valeur est faible, plus on a confiance dans H1 ie plus on
% rejette H0: l'hypothèse de nullié du coefficient

% Régression Ridge
Y = total(:,2);
phi = [cac40(:,2),dowjones(:,2),af(:,2),bp(:,2),danone(:,2),loreal(:,2),michelin(:,2),orange(:,2),vinci(:,2),brut(:,2)];
lam = 2;
theta_ridge = pinv((phi'*phi)+lam*diag(ones(p,1)))*phi'*Y;
% Choix de lambda - Cross validation: ici n = 256: on partitionne en K = 16
% ensembles de 16 éléments chacun
K = 16;
lambda = [-20:0.1:50];
Nu = size(lambda);
Nu = Nu(2);

for u=1:Nu
   for k = 1:K-1
        Y_moins_k = Y;
        phi_moins_k = phi;
        %Pour construire theta_ridge sur toutes les données moins celles de l'esemble k
        for i=1:K
           Y_moins_k(K*(k-1)+i)=[]; 
           phi_moins_k(K*(k-1)+i,:)=[]; 
        end
        %Yj,phij dans k - on ne garde que l'ensemble numéro k
        for j=1:K
            phi_que_k(j,:) = phi(K*(k-1)+j,:); 
            Y_que_k(j)=Y(K*(k-1)+j);
        end
      
        theta_ridge_moins_k =  pinv((phi_moins_k'*phi_moins_k)+lambda(u)*diag(ones(p,1)))*phi_moins_k'*Y_moins_k;
        
        V = Y_que_k'-phi_que_k*theta_ridge_moins_k;
        CV_erreur_sur_k(k) = (1/K)*(V'*V);
   end
   CV_totale(u) = (1/K)*sum(CV_erreur_sur_k);       
end
figure;
plot(lambda,CV_totale)
legend('Erreur de cross-validation en fonciton de lambda');
[CV_totale_min,u_cvmin] = min(CV_totale)
lambda_star = lambda(1) +0.1*u_cvmin;
theta_ridge_star = pinv((phi'*phi)+lambda_star*diag(ones(p,1)))*phi'*Y

