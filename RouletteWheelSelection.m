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
%

function j=RouletteWheelSelection(P)

    r=rand;
    C=cumsum(P);
    j=find(r<=C,1,'first');

end