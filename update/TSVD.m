function [J, f] = TSVD(Q, ref, dim)
    if nargin < 3 || isempty(dim)
        dim = 0;
    end

    Q = shiftdim(Q, dim);
    [~, ~, n3] = size(Q);

    if isscalar(ref)
        ref = repmat(ref, 1, n3);
    end

    if n3 == 2 && isreal(Q)
        [Jplus,fplus]=symmetricSVT(Q(:,:,1)+Q(:,:,2),ref(1));
        [Jminus,fminus]=symmetricSVT(Q(:,:,1)-Q(:,:,2),ref(2));
        J=zeros(size(Q));
        J(:,:,1)=(Jplus+Jminus)/2;
        J(:,:,2)=(Jplus-Jminus)/2;
        if nargout > 1
            f=(fplus+fminus)/2;
        end
        J=shiftdim(J,3-dim);
        return;
    end

    % FFT 
    Qf = fft(Q, [], 3);
    Jf = zeros(size(Qf));
    f = 0;

    for i = 1:n3
        Qi=Qf(:,:,i);
        symmetry_error=norm(Qi-Qi','fro')/(norm(Qi,'fro')+eps);
        if isreal(Qi) && symmetry_error < 1e-12
            Qi=(Qi+Qi')/2;
            [V,e]=eig(Qi,'vector');
            e=real(e);
            s_new=max(abs(e)-ref(i),0);
            weights=sign(e).*s_new;
            Jf(:,:,i)=bsxfun(@times,V,weights')*V';
        else
            [U, S, V] = svd(Qi, 'econ');
            s = diag(S);
            s_new=max(s-ref(i),0);
            Jf(:,:,i)=U*diag(s_new)*V';
        end

        if nargout > 1
            f = f + sum(s_new);
        end
    end

    if nargout > 1
        f = f/n3;
    end

    J = ifft(Jf, [], 3, 'symmetric');
    J = shiftdim(J, 3 - dim);
end

function [J,f]=symmetricSVT(Q,ref)
    Q=(Q+Q')/2;
    [V,e]=eig(Q,'vector');
    e=real(e);
    s=max(abs(e)-ref,0);
    weights=sign(e).*s;
    J=bsxfun(@times,V,weights')*V';
    f=sum(s);
end
