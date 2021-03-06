%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.

function [BestSolDE,DBestSol,BestCostDE] = DE(N, Max_Iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj)

%% Initialization

empty_individual.Position=[];
empty_individual.Cost=[];

BestSolDE.Cost=inf;

pop=repmat(empty_individual,N,1);
% if size(up,1)>1
%     for i=1:dim
%         high=up(i);ll=low(i);
%     end
% end
for i=1:N
    pop(i).Position=rand(1,dim).*(ub-lb)+lb;
    pop(i).Cost=fobj(pop(i).Position);
    
%     if pop(i).Cost<BestSolDE.Cost
%         BestSolDE=pop(i);
%     end
    
end
% Best Solution Ever Found
DBestSol=zeros(1,dim);

BestCostDE=zeros(Max_Iteration,1);

%% DE Main Loop

for it=1:Max_Iteration
    
    for i=1:N
%         for k=1:dim
        
        x=pop(i).Position;
        
        A=randperm(N);
        
        A(A==i)=[];
        
        a=A(1);
        b=A(2);
        c=A(3);
        
        
        % Mutation
        %beta=unifrnd(beta_min,beta_max);
         beta=rand(1,dim).*(beta_max-beta_min)+beta_min;
%         beta=unifrnd(beta_min,beta_max,dim);
        y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);
        y = max(y, lb);
		y = min(y, ub);
%         end 
        % Crossover
        z=zeros(size(x));
        j0=randi([1 numel(x)]);
        for j=1:numel(x)
            if j==j0 || rand<=pCR
                z(j)=y(j);
            else
                z(j)=x(j);
            end
        end
        
        NewSol.Position=z;
%         NewSol.Cost=CostFunction(NewSol.Position);
       NewSol.Cost=fobj(NewSol.Position);
        if NewSol.Cost<pop(i).Cost
            pop(i)=NewSol;
            
            if pop(i).Cost<BestSolDE.Cost
               BestSolDE=pop(i);
            end
        end
        
    end
   % Update Best Solution Ever Found
    DBestSol=pop.Position;
    % Update Best Cost
    BestCostDE(it)=BestSolDE.Cost;
    
%     % Show Iteration Information
%      disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

