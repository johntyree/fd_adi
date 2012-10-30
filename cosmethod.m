% kappa = mean_reversion
% theta = long term mean
% sigma = vol of variance
% rho   = correlation
function P = cosmethod(S, K, r, vol, T, kappa, theta, sigma, rho)

    % tic
    P=COS(S,K,r,vol,T,kappa,theta,sigma,rho)';
    % toc
end


function ret = COS(S,K,r,vol,T,kappa,theta,sigma,rho)
    N = 2^8;
    % Truncation range
    L = 12;
    c1 = r*T + (1 - exp(-kappa * T)) * (theta - vol^2)/(2*kappa) - 0.5*theta*T

    c2 = 1/(8*kappa^3) * (...
        sigma*T*kappa*exp(-kappa*T)*(vol-theta)*(8*kappa*rho - 4*sigma) ...
       + kappa*rho*sigma*(1-exp(-kappa*T))*(16*theta-8*vol)...
       + 2*theta*kappa*T*(-4*kappa*rho*sigma + sigma^2 + 4*kappa^2)
       + sigma^2*((theta - 2*vol)*exp(-2*kappa*T) + theta*(6*exp(-kappa*T)-7) + 2*vol)...
       + 8 * kappa^2 * (vol - theta) * (1-exp(-kappa*T)))

    x = log(S./K);
    a = x+c1-L*sqrt(abs(c2));
    b = x+c1+L*sqrt(abs(c2));
    k=(0:N-1)';
    U = 2/(b-a)*(xi(k,a,b,0,b) - psi(k,a,b,0,b));
    unit= [.5 ones(1,N-1)];
    NPTS = length(S);
    CF_tiled = CF(k*pi/(b-a),S, K, r, vol, T, kappa, theta, sigma, rho) * ones(1,NPTS);

    ret = unit*(CF_tiled.*exp(1i*k*pi*(x-a)/(b-a)).*(U*ones(1, NPTS)));
    ret=K*exp(-r*T).*real(ret);
end

function ret = xi(k,a,b,c,d)
    ret = 1./(1 + (k.*pi / (b-a)).^2)...
        .* (cos(k.*pi.*(d-a)/(b-a)).*exp(d)...
        - cos(k.*pi.*(c-a)/(b-a)) .* exp(c)...
        + k.*pi/(b-a) .* sin(k.*pi.*(d-a)/(b-a)) .* exp(d)...
        - k.*pi/(b-a) .* sin(k.*pi.*(c-a)/(b-a)) .* exp(c));
end

function ret =psi(k,a,b,c,d)
    numerator = (sin(k(2:end)*pi*(d-a) / (b-a)...
                    )...
                 - sin(k(2:end)*pi*(c-a)/(b-a)...
                      )
                ).*(b-a);
    ret = [d-c; numerator ./ (k(2:end)*pi)];
end


function [phi] = CF(w,S,K,r,vol,t,kappa,theta,sigma,rho)
    l_ipnw = kappa - 1i * rho * theta * w;
    D = sqrt((l_ipnw).^2 + (w.^2 + 1i*w)*sigma.^2);
    G = (l_ipnw - D) ./ (l_ipnw + D);
    edt = exp(-D.*t);
    left = exp(1i .* w.*r.*t + vol^2 ./ sigma^2 .* (1-edt) ./ (1 - G.*edt) .* (l_ipnw - D));
    right = exp(kappa.*theta./sigma^2 .* (t.*(l_ipnw-D) - 2.*log((1-G.*edt)./(1-G))));
    phi = left .* right;
end
