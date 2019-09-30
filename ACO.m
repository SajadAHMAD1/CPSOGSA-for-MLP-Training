%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YOEA103
% Project Title: Ant Colony Optimization for Quadratic Assignment Problem
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%
% clc;
% clear;
% close all;
%% Problem Definition
% model=CreateModel();
% CostFunction=@(p) MyCost(p,model);
% nVar=model.n;
function [BestSolACO,BestAnt,BestCostACO] = ACO(N, Max_Iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj)
% [down,up,dim]=benchmark_functions_details(Benchmark_Function_ID);
% [lb,ub,dim,fobj]=Get_Functions_details(Function_name);
% %% ACO Parameters
% MaxIt=500;      % Maximum Number of Iterations
% nAnt=50;        % Number of Ants (Population Size)
% Q=1;
% tau0=10;        % Initial Phromone
% alpha=0.3;      % Phromone Exponential Weight
% rho=0.1;        % Evaporation Rate
%% Initialization
tau=tau0*ones(dim,dim);
% Empty Ant
empty_ant.Tour=[];
empty_ant.Cost=[];

% if size(ub,1)>1
%     for i=1:dim
%         high=ub(i);down=lb(i);
%     end
% end
% Ant Colony Matrix
ant=repmat(empty_ant,N,1);
for i=1:N
%     ant(i).tour=unifrnd(ll,high,dim);
     ant(i).tour=rand(1,N).*(ub-lb)+lb;
%     ant(i).Cost=benchmark_functions(ant(i).tour,Benchmark_Function_ID,dim);
    
%     if ant(i).Cost<BestSolACO.Cost
%         BestSolACO=ant(i);
%     end
end
% Best ant Ever Found
BestAnt=zeros(1,dim);
% Array to Hold Best Cost Values
BestCostACO=zeros(Max_Iteration,1); 
%Best Ant
BestSolACO.Cost=inf;
%% ACO Main Loop
for it=1:Max_Iteration
    
    % Move Ants
     for k=1:N
%         
         ant(k).Tour=[];
        
        for l=1:dim
            
            P=tau(:,l).^alpha;
            
            P(ant(k).Tour)=0;
            
            P=P/sum(P);
            
            j=RouletteWheelSelection(P);
            
            ant(k).Tour=[ant(k).Tour j];
            
        end
        
%         ant(k).Cost=CostFunction(ant(k).Tour);
       ant(k).Cost= fobj(ant(k).Tour);
        if ant(k).Cost<BestSolACO.Cost
            BestSolACO=ant(k);
        end
        
    end
    
    % Update Phromones
    for k=1:N
        
        tour=ant(k).Tour;
        
        for l=1:dim
            
            tau(tour(l),l)=tau(tour(l),l)+Q/ant(k).Cost;
            
        end
        
    end
    
    % Evaporation
    tau=(1-rho)*tau;
    
    % Update Best Solution Ever Found
%     BestAnt=ant(1);
     BestAnt=ant.Tour;
    
    % Store Best Cost
    BestCostACO(it)=BestSolACO.Cost;
    
%     % Show Iteration Information
%     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
%     % Plot Solution
%     figure(1);
%     PlotSolution(BestSolACO.Tour,model);
%     pause(0.01);
    
end
% %% Results
% figure;
% plot(BestCost,'LineWidth',2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;
