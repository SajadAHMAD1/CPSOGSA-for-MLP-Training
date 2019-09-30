%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA113
% Project Title: Biogeography-Based Optimization (BBO) in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

% clc;
% clear;
% close all;
function [BestCost,Best_Hab,BestSol] = bbo( N, Max_Iteration,lb,ub,dim,fobj)
%% Problem Definition

% CostFunction=@(x) Sphere(x);        % Cost Function

% nVar=5;             % Number of Decision Variables
% 
% VarSize=[1 nVar];   % Decision Variables Matrix Size
% 
% VarMin=-10;         % Decision Variables Lower Bound
% VarMax= 10;         % Decision Variables Upper Bound
% [down,up,dim]=benchmark_functions_details(Benchmark_Function_ID);
%% BBO Parameters

% MaxIt=1000;          % Maximum Number of Iterations
VarSize=[1 dim];
% N=50;            % Number of Habitats (Population Size)

KeepRate=0.2;                   % Keep Rate
nKeep=round(KeepRate*N);     % Number of Kept Habitats

nNew=N-nKeep;                % Number of New Habitats

% Migration Rates
mu=linspace(1,0,N);          % Emmigration Rates
lambda=1-mu;                    % Immigration Rates

alpha=0.9;

pMutation=0.1;



%% Initialization
% if size(up,1)==1
%     current_position=rand(dim,N).*(up-down)+down;
         sigma=0.02*(ub-lb);
% end
% if size(up,1)>1
%     for i=1:dim
%         high=up(i);ll=down(i);  
%         sigma=0.02*(high-ll);
%     end
% end
% Empty Habitat
habitat.Position=[];
habitat.Cost= [];
% benchmark_functions(currentX,Benchmark_Function_ID,dim);

% Create Habitats Array
pop=repmat(habitat,N,1);

% Initialize Habitats
for i=1:N
   pop(i).Position= (ub-lb).* rand(1,dim) + lb;
%     pop(i).Position=unifrnd(down,up,VarSize);
%     pop(i).Position=unifrnd(ll,high,VarSize);
    pop(i).Cost=fobj(pop(i).Position);
end

% Sort Population
[~, SortOrder]=sort([pop.Cost]);
pop=pop(SortOrder);

Best_Hab= zeros(1,dim);
% Best Solution Ever Found
BestSol=pop(1);

% Array to Hold Best Costs
BestCost=zeros(Max_Iteration,1);

%% BBO Main Loop

for it=1:Max_Iteration
    
    newpop=pop;
    for i=1:N
        for k=1:dim
            % Migration
            if rand<=lambda(i)
                % Emmigration Probabilities
                EP=mu;
                EP(i)=0;
                EP=EP/sum(EP);
                
                % Select Source Habitat
                j=RouletteWheelSelection(EP);
                
                % Migration
                newpop(i).Position(k)=pop(i).Position(k) ...
                    +alpha*(pop(j).Position(k)-pop(i).Position(k));
                
            end
            
            % Mutation
            if rand<=pMutation
                newpop(i).Position(k)=newpop(i).Position(k)+sigma*randn;
            end
        end
        
        % Apply Lower and Upper Bound Limits
%         newpop(i).Position = max(newpop(i).Position, down);
%         newpop(i).Position = min(newpop(i).Position, up);
%          % Apply Lower and Upper Bound Limits
        newpop(i).Position = max(newpop(i).Position, lb);
        newpop(i).Position = min(newpop(i).Position, ub);
        
        % Evaluation
        newpop(i).Cost=fobj(newpop(i).Position);
    end
    
    % Sort New Population
    [~, SortOrder]=sort([newpop.Cost]);
    newpop=newpop(SortOrder);
    
    % Select Next Iteration Population
    pop=[pop(1:nKeep)
         newpop(1:nNew)];
     
    % Sort Population
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    % Update Best Solution Ever Found
    BestSol=pop(1);
    
    Best_Hab= pop.Position;
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
%     % Show Iteration Information
%     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

% %% Results
% 
% figure;
% %plot(BestCost,'LineWidth',2);
% semilogy(BestCost,'LineWidth',2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;


