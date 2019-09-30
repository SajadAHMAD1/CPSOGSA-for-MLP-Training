%  Traning Feed-forward Neural Networks using CPSOGSA   %
%                                                                   %
%  Developed in MATLAB R2013b                                       %
%                                                                   %
%  programmer: Sajad Ahmad Rather                        %
%                                                                   %
%         e-Mail: sajad.win8@gmail.com                              %
%                                                                   %
% Homepage: https://www.linkedin.com/in/sajad-ahmad-rather-97a398110/   %
%                                                                   %
%   Main paper: Sajad Ahmad Rather, P Shanthi Bala,


clear all 
clc
Q=1;            % ACO Parameter
tau0=10;        % Initial Phromone             (ACO)
alpha=0.3;      % Phromone Exponential Weight  (ACO)
rho=0.1;        % Evaporation Rate             (ACO)
beta_min=0.2;   % Lower Bound of Scaling Factor (DE)
beta_max=0.8;   % Upper Bound of Scaling Factor (DE)
pCR=0.2;        % Crossover Probability         (DE)
Runno=10;

SearchAgents_no=20; % Number of search agents

% classification datasets

% Function_name='F1'; %MLP_XOR dataset
% Function_name='F2'; %MLP_Baloon dataset
%  Function_name='F3'; %MLP_Iris dataset
% Function_name='F4'; %MLP_Cancer dataset
Function_name='F5'; %MLP_Heart dataset

% Function approximation datasets

% Function_name='F6'; %MLP_Sigmoid dataset
% Function_name='F7'; %MLP_Cosine dataset
% Function_name='F8'; %MLP_Sine dataset
% Function_name='F9'; %MLP_Sphere dataset

% Load details of the selected data set
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);

ElitistCheck=1;
min_flag=1;
Rpower=1;
Max_iteration=50;
0; % Maximum numbef of iterations

% 
% if Function_name=='F1' 
% input=  [0 0 0 0 1 1 1 1;0 0 1 1 0 0 1 1;0 1 0 1 0 1 0 1];
% target3=[0 1 1 0 1 0 0 1];
%  Hno=7;
% dim = 5*7+1;                      % Dimension of the problem
%  
%     for i=1:1:Runno
%         Rrate=0;
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions3(i) = gBestScore1;
%          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%                W=Lbest(1:4*Hno);
%                B=Lbest(4*Hno+1:dim);
%                W=PcgCurve(1:4*Hno);
%                B=PcgCurve(4*Hno+1:dim);
%                  W=gBest1(1:4*Hno);
%                  B=gBest1(4*Hno+1:dim);
%                  W=gBest(1:4*Hno);
%                  B=gBest(4*Hno+1:dim);
% W=BestAnt(1:4*Hno);
% B=BestAnt(4*Hno+1:dim);
% W=Best_Hab(1:4*Hno);
% B=Best_Hab(4*Hno+1:dim);
% W=DBestSol(1:4*Hno);
% B=DBestSol(4*Hno+1:dim);
%         for pp=1:8
%             actualvalue=my_simulate(3,Hno,1,W,B,input(:,pp)');
%             if(target3(pp)==1)
%                 if (actualvalue>=0.95)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(target3(pp)==0)
%                 if (actualvalue(1)<0.05)
%                     Rrate=Rrate+1;
%                 end  
%             end
%         end
%          
%         Classification_rate(i)=(Rrate/8)*100;
%   disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),' Mean_Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),' Mean_Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),' Mean_Classification rate =' , num2str(Classification_rate(i)),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),' Mean_Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),' Mean_Classification rate =' , num2str(Classification_rate(i)),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),' Mean_Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),' Mean_Classification rate = ' , num2str(Classification_rate(i)),')']) 
%      end
% %     disp([ num2str(Classification_rate)])
%     A_Classification_rate=mean(Classification_rate);
%     Average= mean(BestSolutions1);
%     StandDP=std(BestSolutions1);
%     Med = median(BestSolutions1); 
%     [BestValueP I] = min(BestSolutions1);
%     [WorstValueP IM]=max(BestSolutions1);
%  end
    
%     
% if Function_name=='F2'
% 
% load baloon.txt
%  x=sortrows(baloon,2);
%  %I2=x(1:150,1:4);
%  I2(:,1)=x(1:20,1);
%  I2(:,2)=x(1:20,2);
%  I2(:,3)=x(1:20,3);
%  I2(:,4)=x(1:20,4);
%  T=x(1:20,5);
%  
% 
% Hno=9;
% dim = 6*9+1;
%  
%    for i=1:1:Runno
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions3(i) = gBestScore1;
%          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         Rrate=0;
%                 W=Lbest(1:45);
%                 B=Lbest(46:55);
%                  W=PcgCurve(1:45);
%                 B=PcgCurve(46:55);
%                   W=gBest1(1:45);
%                   B=gBest1(46:55);
%                 W=gBest(1:45);
%                 B=gBest(46:55);
%         W=BestAnt(1:45);
% B=BestAnt(46:55);
% W=Best_Hab(1:45);
% B=Best_Hab(46:55);
% W=DBestSol(1:45);
% B=DBestSol(46:55);
%         for pp=1:20
%             actualvalue=my_simulate(4,9,1,W,B,I2(pp,:));
%             if(T(pp)==1)
%                 if (actualvalue>=0.95)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(T(pp)==0)
%                 if (actualvalue(1)<0.05)
%                     Rrate=Rrate+1;
%                 end  
%             end
% 
%         end
%         
%         
%       Classification_rate(i,:)=(Rrate/20)*100;
%   disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),' Classification rate =' , num2str(Classification_rate(i)),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),' Classification rate =' , num2str(Classification_rate(i)),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')']) 
%     end
% %     disp([ num2str(Classification_rate)])
%     A_Classification_rate=mean(Classification_rate);
%     Average= mean(BestSolutions);
%     StandDP=std(BestSolutions);
%     Med = median(BestSolutions); 
%     [BestValueP I] = min(BestSolutions);
%     [WorstValueP IM]=max(BestSolutions);
% end
% % % 
%  if Function_name=='F3' 
%     
%     load iris.txt;
%  x=sortrows(iris,2);
%  I2=x(1:150,1:4);
%  H2=x(1:150,1);
%  H3=x(1:150,2);
%  H4=x(1:150,3);
%  H5=x(1:150,4);
%  T=x(1:150,5);
%  I=(I2-0.1)./(7.9-0.1);
%  H2=H2';
%  [xf,PS] = mapminmax(H2);
%  I2(:,1)=xf;
%  
%  H3=H3';
%  [xf,PS2] = mapminmax(H3);
%  I2(:,2)=xf;
%  
%  H4=H4';
%  [xf,PS3] = mapminmax(H4);
%  I2(:,3)=xf;
%  
%  H5=H5';
%  [xf,PS4] = mapminmax(H5);
%  I2(:,4)=xf;
%  Thelp=T;
%  T=T';
%  [yf,PS5]= mapminmax(T);
%  T=yf;
%  T=T';
%  
%     for i=1:1:Runno
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions3(i) = gBestScore1;
%          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         Rrate=0;
%         W=Lbest(1:63);
%         B=Lbest(64:75);
%         W=PcgCurve(1:63);
%         B=PcgCurve(64:75);
%         W=gBest1(1:63);
%         B=gBest1(64:75);
%           W=gBest(1:63);
%           B=gBest(64:75);
%             W=BestAnt(1:63);
%             B=BestAnt(64:75);
%             W=Best_Hab(1:63);
%             B=Best_Hab(64:75);
%             W=DBestSol(1:63);
%             B=DBestSol(64:75);
%         for pp=1:150
%             actualvalue=my_simulate(4,9,3,W,B,I2(pp,:));
%             if(T(pp)==-1)
%                 if (actualvalue(1)>=0.95 && actualvalue(2)<0.05 && actualvalue(3)<0.05)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(T(pp)==0)
%                 if (actualvalue(1)<0.05 && actualvalue(2)>=0.95 && actualvalue(3)<0.05)
%                     Rrate=Rrate+1;
%                 end  
%             end
%             if(T(pp)==1)
%                 if (actualvalue(1)<0.05 && actualvalue(2)<0.05 && actualvalue(3)>=0.95)
%                     Rrate=Rrate+1;
%                 end              
%             end
%         end
%         
%         Classification_rate(i,:)=(Rrate/150)*100;
%   disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),' Classification rate =' , num2str(Classification_rate(i)),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),' Classification rate =' , num2str(Classification_rate(i)),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')']) 
%      end
%     
%     A_Classification_rate=mean(Classification_rate);
%     Average= mean(BestSolutions1);
%     StandDP=std(BestSolutions1);
%     Med = median(BestSolutions1); 
%     [BestValueP I] = min(BestSolutions1);
%     [WorstValueP IM]=max(BestSolutions1);
%     
% end
% % %    
% 
% if Function_name=='F4'
%     
%     load Cancer.txt
%  x=Cancer;
%  %I2=x(1:150,1:4);
%  H2=x(1:699,2:11);
%  for iii=1:699
%      for jjj=1:10
%          H2(iii,jjj)=((H2(iii,jjj)-1)/9);
%      end
%  end
%  I2=H2(1:699,1:9);
%  
%  T=H2(1:699,10);
%  Hno=19;
%  dim=11*19;
%  
%     for i=1:1:Runno
%         Rrate=0;
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions3(i) = gBestScore1;
%          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
% W=Lbest(1:10*Hno);
% B=Lbest(10*Hno+1:dim);
% W=PcgCurve(1:10*Hno);
% B=PcgCurve(10*Hno+1:dim); 
% W=gBest1(1:10*Hno);
% B=gBest1(10*Hno+1:dim);
% W=gBest(1:10*Hno);
% B=gBest(10*Hno+1:dim);
% W=BestAnt(1:10*Hno);
% B=BestAnt(10*Hno+1:dim);
% W=Best_Hab(1:10*Hno);
% B=Best_Hab(10*Hno+1:dim);
% W=DBestSol(1:10*Hno);
% B=DBestSol(10*Hno+1:dim);
%         for pp=600:699
%             actualvalue=my_simulate(9,Hno,1,W,B,I2(pp,:) );
%             if(T(pp)>=0.3 && T(pp)<0.4)
%                 if (abs(actualvalue-0.333333333333333)<0.1)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(T(pp)>=0.1 && T(pp)<0.2)
%                 if (abs(actualvalue-0.111111111111111)<0.1)
%                     Rrate=Rrate+1;
%                 end  
%             end
% 
%         end
%         
%         Classification_rate(1,i)=(Rrate/100)*100;
% %         disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),' Classification rate =' , num2str(Classification_rate(i)),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),' Classification rate =' , num2str(Classification_rate(i)),')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% 
%     end
%     A_Classification_rate=mean(Classification_rate);
%     Average= mean(BestSolutions7);
%     StandDP=std(BestSolutions7);
%     Med = median(BestSolutions7); 
%     [BestValueP I] = min(BestSolutions7);
%     [WorstValueP IM]=max(BestSolutions7);
% end
    
if Function_name=='F5'

load Heart.txt
 x=Heart;
% I2=x(1:150,1:4);
 I2(:,1)=x(1:80,2);
 I2(:,2)=x(1:80,3);
 I2(:,3)=x(1:80,4);
 I2(:,4)=x(1:80,5);
 I2(:,5)=x(1:80,6);
 I2(:,6)=x(1:80,7);
 I2(:,7)=x(1:80,8);
 I2(:,8)=x(1:80,9);
 I2(:,9)=x(1:80,10);
 I2(:,10)=x(1:80,11);
 I2(:,11)=x(1:80,12);
 I2(:,12)=x(1:80,13);
 I2(:,13)=x(1:80,14);
 I2(:,14)=x(1:80,15);
 I2(:,15)=x(1:80,16);
 I2(:,16)=x(1:80,17);
 I2(:,17)=x(1:80,18);
 I2(:,18)=x(1:80,19);
 I2(:,19)=x(1:80,20);
 I2(:,20)=x(1:80,21);
 I2(:,21)=x(1:80,22); 
 I2(:,22)=x(1:80,23);  
 T=x(1:80,1);

 Hno=45;
dim = 24*45+1;    
 
    for i=1:1:Runno

        Rrate=0;
        [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
        BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions3(i) = gBestScore1;
         [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
         BestSolutions4(i) = gBestScore;
        [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
        BestSolutions5(i) = BestSolACO.Cost;
        [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
        BestSolutions6(i) = BestSol.Cost;
        [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
        BestSolutions7(i) = BestSolDE.Cost ;
        W=Lbest(1:23*Hno);
        B=Lbest(23*Hno+1:dim);
% W=PcgCurve(1:23*Hno);
% B=PcgCurve(23*Hno+1:dim); 
% W=gBest1(1:23*Hno);
% B=gBest1(23*Hno+1:dim);
W=gBest(1:23*Hno);
B=gBest(23*Hno+1:dim);
W=BestAnt(1:23*Hno);
B=BestAnt(23*Hno+1:dim);
W=Best_Hab(1:23*Hno);
B=Best_Hab(23*Hno+1:dim);
W=DBestSol(1:23*Hno);
B=DBestSol(23*Hno+1:dim);
        for pp=1:80
            actualvalue=my_simulate(22,Hno,1,W,B,I2(pp,:) );
            if(T(pp)==1)
                if (actualvalue>=0.95)
                    Rrate=Rrate+1;
                end
            end
            if(T(pp)==0)
                if (actualvalue(1)<0.05)
                    Rrate=Rrate+1;
                end  
            end

        end
        
       Classification_rate(i)=(Rrate/80)*100;
 disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),' Classification rate =' , num2str(Classification_rate(i)),')'])  
% disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),' Classification rate =' , num2str(Classification_rate(i)),')'])
% disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
    end
    
 A_Classification_rate=mean(Classification_rate);
    Average= mean(BestSolutions1);
    StandDP=std(BestSolutions1);
    Med = median(BestSolutions1); 
    [BestValueP I] = min(BestSolutions1);
    [WorstValueP IM]=max(BestSolutions1);
end

% if Function_name=='F6'  %% Sigmoid
%     
%     Hnode=15;
% dim = 3*Hnode+1;
%  
% %for test 3 times more than the training samples
% %   xf1=[0:0.01:pi];
% %   yf1=sin(2.*xf1);
% %   yf1=yf1.*exp(-xf1);
%  xf1=[-3:0.05:3];
% %  yf1=sin(2.*xf1);
% %  yf1=yf1.*exp(-xf1);
%  %yf1=xf1.^2;
% %yf1=xf1.^4-6.*xf1.^2+3;
% yf1=sigmf(xf1,[1 0]);
% 
% %   xf1=[-2*pi:0.05:2*pi];
% %   yf1=sin(2.*xf1);
%   %yf1=yf1.*exp(-xf1);
%  yNN=zeros(1,10);
%  [xf,PS] = mapminmax(xf1);
%  [yf,PS2]= mapminmax(yf1);
%   [M N]=size(xf);
%   test_error=zeros(1,Runno);
%     for i=1:1:Runno
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions3(i) = gBestScore1;
%          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         W=1:2*Hnode;
%         B=2*Hnode+1:3*Hnode+1;
% W=PcgCurve(1:2*Hnode);
% B=PcgCurve(2*Hnode+1:3*Hnode+1); 
% W=gBest1(1:2*Hnode);
% B=gBest1(2*Hnode+1:3*Hnode+1);
% W=gBest(1:2*Hnode);
% B=gBest(2*Hnode+1:3*Hnode+1);
% W=BestAnt(1:2*Hnode);
% B=BestAnt(2*Hnode+1:3*Hnode+1);
% W=Best_Hab(1:2*Hnode);
% B=Best_Hab(2*Hnode+1:3*Hnode+1);
% W=DBestSol(1:2*Hnode);
% B=DBestSol(2*Hnode+1:3*Hnode+1);
% 
%         for pp=1:N
%             yNN(pp)=my_simulate(1,15,1, W,B,xf(pp));
%         end       
% 
%  disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O), ')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore), ')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')'])
%     end
% %      A_Classification_rate=mean(Classification_rate);
%     Average= mean(BestSolutions1);
%     StandDP=std(BestSolutions1);
%     Med = median(BestSolutions1); 
%     [BestValueP I] = min(BestSolutions1);
%     [WorstValueP IM]=max(BestSolutions1);
% end

   
% if Function_name=='F7' %% Cosine
%  
% Hnode=15;
% dim = 3*Hnode+1;
%  
% %for test 3 times more than the training samples
% %   xf1=[0:0.01:pi];
% %   yf1=sin(2.*xf1);
% %   yf1=yf1.*exp(-xf1);
%  %xf1=[-3:0.05:3];
% %  yf1=sin(2.*xf1);
% %  yf1=yf1.*exp(-xf1);
%  %yf1=xf1.^2;
% %yf1=xf1.^4-6.*xf1.^2+3;
% 
%   xf1=[1.25:0.04:2.75];
%   yf1=power(cos(xf1.*pi/2),7);
%   %yf1=yf1.*exp(-xf1);
%  yNN=zeros(1,10);
%  [xf,PS] = mapminmax(xf1);
%  [yf,PS2]= mapminmax(yf1);
%   [M N]=size(xf);
%   test_error=zeros(1,Runno);
%     for i=1:1:Runno
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%           [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions3(i) = gBestScore1;
%          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         
%         W= Lbest(1:2*Hnode);
%         B=Lbest(2*Hnode+1:3*Hnode+1);
% W=PcgCurve(1:2*Hnode);
% B=PcgCurve(2*Hnode+1:3*Hnode+1); 
% W=gBest1(1:2*Hnode);
% B=gBest1(2*Hnode+1:3*Hnode+1);
% W=gBest(1:2*Hnode);
% B=gBest(2*Hnode+1:3*Hnode+1);
% W=BestAnt(1:2*Hnode);
% B=BestAnt(2*Hnode+1:3*Hnode+1);
% W=Best_Hab(1:2*Hnode);
% B=Best_Hab(2*Hnode+1:3*Hnode+1);
% W=DBestSol(1:2*Hnode);
% B=DBestSol(2*Hnode+1:3*Hnode+1);
%         for pp=1:N
%             yNN(pp)=my_simulate(1,15,1, W,B,xf(pp));
%         end
% %           
% % disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest)  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O), ')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore), ')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')'])
%     end
% %    A_Classification_rate=mean(Classification_rate);
%     Average= mean(BestSolutions7);
%     StandDP=std(BestSolutions7);
%     Med = median(BestSolutions7); 
%     [BestValueP I] = min(BestSolutions7);
%     [WorstValueP IM]=max(BestSolutions7);
% end

if Function_name=='F8' %%Sine

Hnode=15;
dim = 3*Hnode+1;
 
%for test 3 times more than the training samples
%   xf1=[0:0.01:pi];
%   yf1=sin(2.*xf1);
%   yf1=yf1.*exp(-xf1);
 %xf1=[-3:0.05:3];
%  yf1=sin(2.*xf1);
%  yf1=yf1.*exp(-xf1);
 %yf1=xf1.^2;
%yf1=xf1.^4-6.*xf1.^2+3;

  xf1=[-2*pi:0.05:2*pi];
  yf1=sin(2.*xf1);
  %yf1=yf1.*exp(-xf1);
 yNN=zeros(1,10);
 [xf,PS] = mapminmax(xf1);
 [yf,PS2]= mapminmax(yf1);
  [M N]=size(xf);
  test_error=zeros(1,Runno);
    for i=1:1:Runno
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%           [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%           BestSolutions3(i) = gBestScore1;
%         [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
        [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
        BestSolutions(i) = BestSolDE.Cost ;
% 
%         W=Lbest(1:2*Hnode);
%         B= Lbest(2*Hnode+1:3*Hnode+1);
% W=PcgCurve(1:2*Hnode);
% B=PcgCurve(2*Hnode+1:3*Hnode+1); 
% W=gBest1(1:2*Hnode);
% B=gBest1(2*Hnode+1:3*Hnode+1);
% W=gBest(1:2*Hnode);
% B=gBest(2*Hnode+1:3*Hnode+1);
% W=BestAnt(1:2*Hnode);
% B=BestAnt(2*Hnode+1:3*Hnode+1);
% W=Best_Hab(1:2*Hnode);
% B=Best_Hab(2*Hnode+1:3*Hnode+1);
W=DBestSol(1:2*Hnode);
B=DBestSol(2*Hnode+1:3*Hnode+1);
        
        for pp=1:N
            yNN(pp)=my_simulate(1,15,1, W,B,xf(pp));
        end
%           
%         disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O), ')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore), ')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')'])
    end
% A_Classification_rate=mean(Classification_rate);
    Average= mean(BestSolutions);
    StandDP=std(BestSolutions);
    Med = median(BestSolutions); 
    [BestValueP I] = min(BestSolutions);
    [WorstValueP IM]=max(BestSolutions);
end


% if Function_name=='F9'  %% Sphere
% 
% Hnode=15;
% dim = 4*Hnode+1
%  
% %for test 3 times more than the training samples
% %   xf1=[0:0.01:pi];
% %   yf1=sin(2.*xf1);
% % %   yf1=yf1.*exp(-xf1);
% %  xf1=[-2:0.05:2];
% % %  yf1=sin(2.*xf1);
% % %  yf1=yf1.*exp(-xf1);
% %  yf1=xf1.^2;
% % %yf1=xf1.^4-6.*xf1.^2+3;
% %yf1=sigmf(xf1,[1 0]);
% 
%  xf1=[-2:0.05:2];
%  yf1=xf1.^2; 
%   [xf1,yf1] = meshgrid(-2:.1:2);
% % yf1=xf1.^4-6.*xf1.^2+3;
% %yf1=sigmf(xf1,[1 0]);
%    [M N]=size(xf1);
% for i=1:M
%     for j=1:N
%         L=[xf1(i,j) yf1(i,j)];
%         zf1(i,j)=sum(L.^2);
%     end
% end
% 
%    
%  [xf,PS] = mapminmax(xf1);
%  [yf,PS2]= mapminmax(yf1);
%  [zf,PS3]= mapminmax(zf1);
%   
%   
%   zNN=zeros(1,10);
%   test_error=zeros(1,Runno);
%     for i=1:Runno
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
% %         BestSolutions(i) = Fbest;
% %         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions(i) = GBEST.O;
% %           [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %           BestSolutions(i) = gBestScore1;
% %         [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions1(i) = gBestScore;
%         for ww=1:3*Hnode
%             W(ww)=gBest1(i,ww);
%         end
%         for bb=3*Hnode+1:4*Hnode+1
%             B(bb-3*Hnode)=gBest1(i,bb);
%         end
%         
% %         for pp=1:N
% %             zNN(pp)=my_simulate(W,B,xf(pp),Hnode);
% %         end
% for ii=1:M
%     for jj=1:N
%         L=[xf1(ii,jj) yf1(ii,jj)];
%          zNN(ii,jj)=my_simulate_2_inputs(W,B,xf1(ii,jj),yf1(i,jj),Hnode);
%     end
% end
%         figure
%         set(axes,'FontName','Times New Roman');
%         
%         hold on
%         grid on;
%         xfDenorm = mapminmax('reverse',xf,PS); 
%         yfDenorm = mapminmax('reverse',yf,PS2);
%         zfDenorm = mapminmax('reverse',zNN,PS3);
%         test_error(1,i)=test_error(1,i)+sum(sum(abs( zfDenorm- zf1 )));  
%         %surfc(xf1,yf1,zf1);
%         %colormap('Summer');
%         surfc(xfDenorm,yfDenorm,zfDenorm);
%         view([-38,30])
%         
%         colormap('Winter');
%         plot(xf1,yf1,'DisplayName','Real curve','Color','b');
%         plot(xfDenorm,yfDenorm,'DisplayName','Approximated curve','Marker','.','LineStyle','-','Color','r');
%         eqtext = '$$sin(2x)e^{-x}$$'; 
%        
%         name='PSOGSA';
%   
%         
%         title([['\fontsize{12}\it ', name]],'FontName','Times New Roman');
%         xlabel('\fontsize{12}\it X');
%         ylabel('\fontsize{12}\it Y');
%         zlabel('\fontsize{12}\it Z');
%         %legend('toggle');
%         set(legend,'FontAngle','italic','FontName','Times New Roman')       
% disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% 
%     end
% %     A_Classification_rate=mean(Classification_rate);
%     Average= mean(BestSolutions);
%     StandDP=std(BestSolutions);
%     Med = median(BestSolutions); 
%     [BestValueP I] = min(BestSolutions);
%     [WorstValueP IM]=max(BestSolutions);
% end



% figure
%         set(axes,'FontName','Times New Roman');
%         
%        hold on
%         grid on;
%         xfDenorm = mapminmax('reverse',xf,PS); 
%         yfDenorm = mapminmax('reverse',yNN,PS2);
%         test_error(1,i)=test_error(1,i)+sum(abs( yfDenorm- yf1 ));
%         A_Test_Error=mean(test_error);
%         plot(xf1,yf1,'DisplayName','Real curve','Color','b');
%         plot(xfDenorm,yfDenorm,'DisplayName','Approximated curve','Marker','.','LineStyle','-','Color','r');
%         %eqtext = '$$sin(2x)e^{-x}$$'; 
%       name='DE'
%         
%         
%         title([['\fontsize{12}\it ', name]],'FontName','Times New Roman');
%         xlabel('\fontsize{12}\it X');
%         ylabel('\fontsize{12}\it Y');
%         legend('toggle');
%         set(legend,'FontAngle','italic','FontName','Times New Roman') 


% figure
%         set(axes,'FontName','Times New Roman');
%         
%         hold on
%         grid on;
%         xfDenorm = mapminmax('reverse',xf,PS); 
%         yfDenorm = mapminmax('reverse',yNN,PS2);
%         test_error(1,i)=test_error(1,i)+sum(abs( yfDenorm- yf1 )); 
%         A_Test_Error=mean(test_error);
%         plot(xf1,yf1,'DisplayName','Real curve','Color','b');
%         plot(xfDenorm,yfDenorm,'DisplayName','Approximated curve','Marker','.','LineStyle','-','Color','r');
%         %eqtext = '$$sin(2x)e^{-x}$$'; 
%         
%         name='DE'
% 
%         title([['\fontsize{12}\it ', name]],'FontName','Times New Roman');
%         xlabel('\fontsize{12}\it X');
%         ylabel('\fontsize{12}\it Y');
%         legend('toggle');
%         set(legend,'FontAngle','italic','FontName','Times New Roman') 


        
disp(['Best=',num2str( BestValueP)])
disp(['Worst=',num2str(WorstValueP)])
disp(['Average=',num2str( Average)])
disp(['Standard_Deviation=',num2str( StandDP)])
disp(['Median=',num2str(Med)])
% % disp(['Mean_Test_Error = ' , num2str(A_Test_Error)])


  figure
 semilogy(1:Max_iteration,BestChart,'DisplayName','GSA','Color','g','Marker','o','LineStyle','-','LineWidth',2,...
        'MarkerEdgeColor','g','MarkerFaceColor',[.49 1 .63],'MarkerSize',5);
   hold on
%       semilogy(PcgCurve,'DisplayName','PSO','Color','c','Marker','square','LineStyle','-','LineWidth',2,...
%        'MarkerEdgeColor','c','MarkerFaceColor',[.49 1 .63],'MarkerSize',5);
%    semilogy(1:Max_iteration,GlobalBestCost1,'DisplayName','PSOGSA','Color','m','Marker','<','LineStyle','-','LineWidth',2,...
%        'MarkerEdgeColor','m','MarkerFaceColor',[.49 1 .63],'MarkerSize',5);
   semilogy(1:Max_iteration,GlobalBestCost,'DisplayName','CPSOGSA', 'Color', 'r','Marker','diamond','LineStyle','-','LineWidth',2,...
       'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63],'MarkerSize',5);
semilogy(1:Max_iteration,BestCostACO,'DisplayName','ACO','Color','c','Marker','square','LineStyle','-','LineWidth',2,...
    'MarkerEdgeColor','c','MarkerFaceColor',[.49 1 .63],'MarkerSize',5);
   semilogy(1:Max_iteration,BestCost,'DisplayName','BBO','Color','b','Marker','*','LineStyle','-','LineWidth',2,...
       'MarkerEdgeColor','b','MarkerFaceColor',[.49 1 .63],'MarkerSize',5);
   semilogy(1:Max_iteration,BestCostDE,'DisplayName','DE','Color','y','Marker','+','LineStyle','-','LineWidth',2,...
       'MarkerEdgeColor','y','MarkerFaceColor',[.49 1 .63],'MarkerSize',5);
%    
% % % % %   title ('\fontsize{12}\bf XOR Dataset');
% % % %    title ('\fontsize{12}\bf Baloon Dataset');
% % % %   title ('\fontsize{12}\bf Iris Dataset');
% % % %   title ('\fontsize{12}\bf Cancer Dataset');
 title ('\fontsize{12}\bf Heart Dataset');
% title ('\fontsize{12}\bf Sigmoid Dataset');
% title ('\fontsize{12}\bf Cosine Dataset');
% title ('\fontsize{12}\bf Sine Dataset');
xlabel('\fontsize{12}\bf Iteration');
ylabel('\fontsize{12}\bf log(MSE)');
% legend('\fontsize{10}\bf GSA','\fontsize{10}\bf PSO','\fontsize{10}\bf PSOGSA','\fontsize{10}\bf CPSOGSA','\fontsize{10}\bf ACO','\fontsize{10}\bf BBO','\fontsize{10}\bf DE',1);
legend('\fontsize{10}\bf GSA','\fontsize{10}\bf CPSOGSA','\fontsize{10}\bf ACO','\fontsize{10}\bf BBO','\fontsize{10}\bf DE',1);
% legend('\fontsize{10}\bf DE',1);
axis tight
box on


% Wilcoxon rank sum test

% disp(' Wilcoxon RankSum Test ')
%  [p,h]= signrank(BestSolutions4,BestSolutions5)


 
 %%BoxPlot
 
%  boxplot([BestSolutions1',BestSolutions2',BestSolutions3',BestSolutions4',BestSolutions5',BestSolutions6',BestSolutions7'],...
%  {'GSA','PSO','PSOGSA','CPSOGSA','ACO','BBO','DE'})
% %  boxplot([BestSolutions1',BestSolutions4',BestSolutions5',BestSolutions6',BestSolutions7'],...
% %  {'GSA','CPSOGSA','ACO','BBO','DE'})
% title ('\fontsize{12}\bf Sigmoid Dataset');
%   xlabel('\fontsize{12}\bf Algorithm');
%   ylabel('\fontsize{12}\bf Best Fitness Value');
