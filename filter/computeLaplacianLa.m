function La = computeLaplacianLa(obj)
%
%
%   
    
    Z=max(real((obj.Z+obj.Z')/2),0);
    Z(1:size(Z,1)+1:end)=0;
    degree=sum(Z,2);
    degree_inv_sqrt=1./sqrt(degree+obj.epsilon_D);
    normalized_Z=bsxfun(@times,bsxfun(@times,Z,degree_inv_sqrt), ...
        degree_inv_sqrt');
    La=eye(size(Z))-normalized_Z;
 
end
