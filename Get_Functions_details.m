%  Traning Feed-forward Neural Networks using Grey Wolf Optimizer   %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili,How effective is the Grey Wolf         %
%               optimizer in training multi-layer perceptrons       %
%              Applied Intelligece, in press, 2015,                 %
%               http://dx.doi.org/10.1007/s10489-014-0645-7         %
%                                                                   %

% This function containts full information and implementations of the
% datasets

% lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
% up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]
% dim is the number of variables (dimension of the problem)

function [lb,ub,dim,fobj] = Get_Functions_details(F)


switch F       
   case 'F1'
        fobj=@MLP_XOR;
        lb=-10;
        ub=10;
        dim=36;
        
    case 'F2'
        fobj = @MLP_Baloon;
        lb=-10;
        ub=10;
        dim=55;   
        
    case 'F3'
        fobj=@MLP_Iris;
        lb=-10;
        ub=10;
        dim=75;
        
    case 'F4'
        fobj=@MLP_Cancer;
        lb=-10;
        ub=10;
        dim=209;
        
     case 'F5'
        fobj=@MLP_Heart;
        lb=-10;
        ub=10;
        dim=1081;       
        
     case 'F6'
        fobj=@MLP_Sigmoid;
        lb=-10;
        ub=10;
        dim=46; 
        
     case 'F7'
        fobj=@MLP_Cosine;
        lb=-10;
        ub=10;
        dim=46;    
        
     case 'F8'
         fobj=@MLP_Sine;
         lb=-10;
         ub=10;
         dim=46;
        
     case 'F9'
         fobj=@MLP_Sphere;
         lb=-10;
         ub=10;
         dim=61;
         
end

end

function o=MLP_Baloon(solution)

load baloon.txt
 x=sortrows(baloon,2);
 %I2=x(1:150,1:4);
 I2(:,1)=x(1:20,1);
 I2(:,2)=x(1:20,2);
 I2(:,3)=x(1:20,3);
 I2(:,4)=x(1:20,4);
 T=x(1:20,5);

Hno=9;
dim = 6*9+1;                      % Dimension of the problem

   o  = 0;
        for ww=1:45
            W(ww)=solution(1,ww);
        end
        for bb=46:55
            B(bb-45)=solution(1,bb);
        end        
        fitness=0;
        for pp=1:20
            actualvalue=my_simulate(4,9,1,W,B,I2(pp,:));

                fitness=fitness+(T(pp)-actualvalue)^2;

        end
        fitness=fitness/20;            
        o=fitness;
end

function o=MLP_Iris(solution)

load iris.txt
 x=sortrows(iris,2);
 %I2=x(1:150,1:4);
 H2=x(1:150,1);
 H3=x(1:150,2);
 H4=x(1:150,3);
 H5=x(1:150,4);
 T=x(1:150,5);
 %I=(I2-0.1)./(7.9-0.1)
 H2=H2';
 [xf,PS] = mapminmax(H2);
 I2(:,1)=xf;
 
 H3=H3';
 [xf,PS2] = mapminmax(H3);
 I2(:,2)=xf;
 
 H4=H4';
 [xf,PS3] = mapminmax(H4);
 I2(:,3)=xf;
 
 H5=H5';
 [xf,PS4] = mapminmax(H5);
 I2(:,4)=xf;
 Thelp=T;
 T=T';
 [yf,PS5]= mapminmax(T);
 T=yf;
 T=T';
 

        for ww=1:63
            W(ww)=solution(1,ww);
        end
        for bb=64:75
            B(bb-63)=solution(1,bb);
        end        
        fitness=0;
        for pp=1:150
            actualvalue=my_simulate(4,9,3,W,B,I2(pp,:));
            if(T(pp)==-1)
                fitness=fitness+(1-actualvalue(1))^2;
                fitness=fitness+(0-actualvalue(2))^2;
                fitness=fitness+(0-actualvalue(3))^2;
            end
            if(T(pp)==0)
                fitness=fitness+(0-actualvalue(1))^2;
                fitness=fitness+(1-actualvalue(2))^2;
                fitness=fitness+(0-actualvalue(3))^2;   
            end
            if(T(pp)==1)
                fitness=fitness+(0-actualvalue(1))^2;
                fitness=fitness+(0-actualvalue(2))^2;
                fitness=fitness+(1-actualvalue(3))^2;              
            end
        end
        fitness=fitness/150;
        o=fitness;
end

function o=MLP_XOR(solution)

input=  [0 0 0 0 1 1 1 1;0 0 1 1 0 0 1 1;0 1 0 1 0 1 0 1];
target3=[0 1 1 0 1 0 0 1];

 Hno=7;
dim = 5*7+1;                      % Dimension of the problem

    
        fitness=0;
        for indexi=1:4*Hno
        W(indexi)=solution(1,indexi);
    end
    
        for indexi=4*Hno+1:5*Hno+1
        B(indexi-4*Hno)=solution(1,indexi);
        end
        
        for pp=1:8
            actualvalue=my_simulate(3,7,1,W,B,input(:,pp)');
            fitness=fitness+(target3(pp)-actualvalue)^2;
        end
        

        fitness=fitness/8;         
        o=fitness;

end

function o=MLP_Cancer(solution)
load Cancer.txt
 x=Cancer;
 %I2=x(1:150,1:4);
 H2=x(1:699,2:11);
 for iii=1:699
     for jjj=1:10
         H2(iii,jjj)=((H2(iii,jjj)-1)/9);
     end
 end
 I2=H2(1:699,1:9);
 
 T=H2(1:699,10);
 Hno=19;
 dim=11*19;


        for ww=1:10*Hno
            W(ww)=solution(1,ww);
        end
        for bb=10*Hno+1:dim
            B(bb-(10*Hno))=solution(1,bb);
        end
        fitness=0;
        for pp=1:599
            actualvalue=my_simulate(9,Hno,1,W,B,I2(pp,:));

            fitness=fitness+(T(pp)-actualvalue)^2;

        end
        fitness=fitness/599;
        o=fitness;

end


function o=MLP_Heart(solution)
 load Heart.txt
 x=Heart;
 %I2=x(1:150,1:4);
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
dim = 24*45+1;                      % Dimension of the problem


       
        for ww=1:23*Hno
            W(ww)=solution(1,ww);
        end
        for bb=23*Hno+1:dim
            B(bb-(23*Hno))=solution(1,bb);
        end        
        fitness=0;
        for pp=1:80
            actualvalue=my_simulate(22,Hno,1,W,B,I2(pp,:) );

            fitness=fitness+(T(pp)-actualvalue)^2;

        end
        fitness=fitness/80;    
        o=fitness;
        
end


function o=MLP_Sigmoid(solution)

  xf1=[-3:0.1:3];
% yf1=xf1.^4-6.*xf1.^2+3;
yf1=sigmf(xf1,[1 0]);
 
 [xf,PS] = mapminmax(xf1);
 [yf,PS2]= mapminmax(yf1);
  [M N]=size(xf);


 Hnode=15;
 dim = 3*Hnode+1;                      % Dimension of the problem


        fitness=0;
       for ww=1:2*Hnode
            W(ww)=solution(1,ww);
        end
        for bb=2*Hnode+1:3*Hnode+1
            B(bb-2*Hnode)=solution(1,bb);
        end     
        
        for pp=1:N
            actualvalue=my_simulate(1,15,1,W,B,xf(pp));
            %actualvalueDenorm = mapminmax('reverse',actualvalue,PS);
            fitness=fitness+(yf(pp)-actualvalue)^2;
        end

        fitness=fitness/N;        
               
        o=fitness;
end


function o=MLP_Cosine(solution)

 xf1=[1.25:0.05:2.75];
  yf1=power(cos(xf1.*pi/2),7);
  %yf1=yf1.*exp(-xf1);

%  xf1=[-3:0.1:3];
% yf1=xf1.^4-6.*xf1.^2+3;
 
 [xf,PS] = mapminmax(xf1);
 [yf,PS2]= mapminmax(yf1);
  [M N]=size(xf);


 Hnode=15;
dim = 3*Hnode+1;                      % Dimension of the problem

        fitness=0;
       for ww=1:2*Hnode
            W(ww)=solution(1,ww);
        end
        for bb=2*Hnode+1:3*Hnode+1
            B(bb-2*Hnode)=solution(1,bb);
        end     
        
        for pp=1:N
            actualvalue=my_simulate(1,15,1,W,B,xf(pp));
            %actualvalueDenorm = mapminmax('reverse',actualvalue,PS);
            fitness=fitness+(yf(pp)-actualvalue)^2;
        end

        fitness=fitness/N;        
               
        o=fitness;
end


function o=MLP_Sine(solution)

 xf1=[-2*pi:0.1:2*pi];
  yf1=sin(2.*xf1);
  %yf1=yf1.*exp(-xf1);

%  xf1=[-3:0.1:3];
% yf1=xf1.^4-6.*xf1.^2+3;
 
 [xf,PS] = mapminmax(xf1);
 [yf,PS2]= mapminmax(yf1);
  [M N]=size(xf);


 Hnode=15;
dim = 3*Hnode+1;                      % Dimension of the problem

        fitness=0;
       for ww=1:2*Hnode
            W(ww)=solution(1,ww);
        end
        for bb=2*Hnode+1:3*Hnode+1
            B(bb-2*Hnode)=solution(1,bb);
        end     
        
        for pp=1:N
            actualvalue=my_simulate(1,15,1,W,B,xf(pp));
            %actualvalueDenorm = mapminmax('reverse',actualvalue,PS);
            fitness=fitness+(yf(pp)-actualvalue)^2;
        end

        fitness=fitness/N;        
               
        o=fitness;
end

function o=MLP_Sphere(solution)

[xf1,yf1] = meshgrid(-2:.2:2);
[M, N]=size(xf1);
for i=1:M
    for j=1:N
        L=[xf1(i,j) yf1(i,j)];
        zf1(i,j)=sum(L.^2);
    end
end

   
 [xf,PS] = mapminmax(xf1);
 [yf,PS2]= mapminmax(yf1);
 [zf,PS3]= mapminmax(zf1);

  


 Hnode=15;
dim = 4*Hnode+1;                      % Dimension of the problem

        fitness=0;
       for ww=1:3*Hnode
            W(ww)=solution(1,ww);
        end
        for bb=3*Hnode+1:4*Hnode+1
            B(bb-3*Hnode)=solution(1,bb);
        end     
        
for i=1:M
    for j=1:N
        L=[xf1(i,j) yf1(i,j)];
        actualvalue=my_simulate_2_inputs(W,B,xf1(i,j),yf1(i,j),Hnode);
        actualvalueDenorm = mapminmax('reverse',actualvalue,PS);
        fitness=fitness+(zf(i,j)-actualvalue)^2;
    end
end

        fitness=fitness/N;        
               
        o=fitness;
end