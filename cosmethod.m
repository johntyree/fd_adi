function P = cosmethod(S, K, r, sigma, T, nu, theta)
    % S = 100; K = 90; T = .1; r = .1; sigma = 0.12; nu = 0.2; theta = -.14;

    tic
    P=COS(S,K,T,r,sigma,nu,theta)';
    toc
end


function ret = COS(S,K,T,r,sigma,nu,theta)
    N = 2^8;
    % Truncation rane
    L = 10;
    c1 = (r+theta)*T;
    c2 = (sigma^2+nu*theta^2)*T;
    c4 = 3*(sigma^4*nu+2*theta^4*nu^3+4*sigma^2*theta^2*nu^2)*T;
    a = c1-L*sqrt(c2+sqrt(c4));
    b = c1+L*sqrt(c2+sqrt(c4));
    x = log(S./K);
    [S,K]
    k=(0:N-1)';
    U = 2/(b-a)*(xi(k,a,b,0,b) -psi(k,a,b,0,b));
    unit= [.5 ones(1,N-1)];
    CFVG_tiled = CFVarianceGamma(k*pi/(b-a),S, K, T, r, sigma, nu, theta) *ones(1,length(K));
    x-a
    % exp(1i*k*pi*(x-a)/(b-a))

    ret = unit*(CFVG_tiled.*exp(1i*k*pi*(x-a)/(b-a)).*(U*ones(1,length(K))));
    % ret = unit
    % *(CFVarianceGamma(k*pi/(b-a),S, K, T, r, sigma, nu, theta)*ones(1,length(K))
    % .*exp(1i*k*pi*(x-a)/(b-a))
    % .*(U*ones(1,length(K))));
    ret=K*exp(-r*T).*real(ret);
% ret=K*exp(-r*T)*real(unit*(CFVarianceGamma(k*pi/(b-a),S, K, T, r, sigma, nu, theta).*exp(1i*k*pi.*(x-a)/(b-a)).*U));
end

function ret = xi(k,a,b,c,d)
    ret =1./(1+(k*pi/(b-a)).^2).*(cos(k*pi*(d-a)/(b-a)).*exp(d)...
    -cos(k*pi*(c-a)/(b-a))*exp(c)+k*pi/(b-a).*sin(k*pi*(d-a)/(b-a))*exp(d)...
    -k*pi/(b-a).*sin(k*pi*(c-a)/(b-a))*exp(c));
end

function ret =psi(k,a,b,c,d)
    ret = [d-c;(sin(k(2:end)*pi*(d-a)/(b-a))-sin(k(2:end)*pi*(c-a)/(b-a))).*(b-a)./(k(2:end)*pi)];
end


function [phi] = CFVarianceGamma(u,S,K,t,r,sigma,nu,theta)
    omega = log(1 - theta*nu - sigma^2*nu/2)/nu;
    phi = exp(u*(r + omega)*t*1i);
    (1-1i*theta*nu*u +sigma^2*nu*u.^2/ 2);
    foo = (1-1i*theta*nu*u +sigma^2*nu*u.^2/ 2).^(-t/nu);
    phi = phi.*foo;
end
