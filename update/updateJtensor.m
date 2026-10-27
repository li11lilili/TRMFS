function [J,fTSVD]=updateJtensor(obj)
    ref = 1 / obj.mu;
    T = obj.A_tensor+(obj.Q_tensor ./ obj.mu);
    [J,fTSVD] = TSVD(T,ref);
end