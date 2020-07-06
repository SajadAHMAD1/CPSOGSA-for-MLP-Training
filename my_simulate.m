%  Multi-layer Perceptron (MLP) Training using CPSOGSA 
%
% Citation
% Rather, S.A. and Bala, P.S. (2020), "A hybrid constriction coefficient-based particle swarm optimization and gravitational search algorithm for training multi-layer perceptron", 
% International Journal of Intelligent Computing and Cybernetics, Vol. 13 No. 2, pp. 129-165. https://doi.org/10.1108/IJICC-09-2019-0105  
%
%  Developed in MATLAB R2013b                                       %
%                                                                   %
%  Developer and programmer: Sajad Ahmad Rather                        %
%                                                                   %
%         E_Mail: sajad.win8@gmail.com                              %
%                                                                   %
% Homepage: https://www.linkedin.com/in/sajad-ahmad-rather-97a398110/   %
%                                                                   %

% This function simulates the MLP

function o=my_simulate(Ino,Hno,Ono,W,B,x)
h=zeros(1,Hno);
o=zeros(1,Ono);
index=-1;

for i=1:Hno

    index=index+1;    
    ssum=0;
    for j=1:size(x,2)
        ssum= ssum+x(1,j)*W(index*Ino+j);
    end
    h(i)=My_sigmoid(ssum+B(i));
end

k=size(x,2);

for j=1:Hno
    o=o+(h(j)*W(k*Hno+j));
end

for k=1:Ono
    o(k)=My_sigmoid(o(k)+B(Ino+1));
end


