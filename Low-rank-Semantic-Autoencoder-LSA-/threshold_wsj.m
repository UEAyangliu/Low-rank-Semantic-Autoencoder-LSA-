function[F]=threshold_wsj(X,tau)
    F = sign(X).*max(abs(X)-tau,0);
end