% kappa = mean_reversion
% theta = long term mean
% sigma = vol of variance
% rho   = correlation
function P = cosmethodjohnedit(S, K, r, vol, T, kappa, theta, sigma, rho)

    % tic
    P=COS(S,K,r,vol,T,kappa,theta,sigma,rho)';
    % toc
end


function ret = COS(S,K,r,vol,T,kappa,theta,sigma,rho)
    N = 2^8;
    
    L = 12;
    x= log(S./K);  %scaled log-asset price
    mu=r;

    %Cumulants
    c1=mu*T+(1-exp(-kappa*T))*(theta-vol)/(2*kappa)-0.5*theta*T;
    p1=sigma*T*kappa*exp(-kappa*T)*(vol-theta)*(8*kappa*rho-4*sigma);
    p2=kappa*rho*sigma*(1-exp(-kappa*T))*(16*theta-8*vol);
    p3=2*theta*kappa*T*(-4*kappa*rho*sigma+sigma^2+4*kappa^2);
    p4=sigma^2*((theta-2*vol)*exp(-2*kappa*T)+theta*(6*exp(-kappa*T)-7)+2*vol);
    p5=8*kappa^2*(vol-theta)*(1-exp(-kappa*T));
    c2=1/(8*kappa^3)*(p1+p2+p3+p4+p5);

    [S,K,r,vol,T,kappa,theta,sigma,rho]
    p1
    p2
    p3
    p4
    p5
    c1
    c2
    %Interval [a,b]
    a=x+c1-L*sqrt(abs(c2));
    b=x+c1+L*sqrt(abs(c2));
    k=0:N-1; 
    omega=k'*pi/(b-a);  

    %Characteristic function
    cf=ChFHeston(omega,vol,theta,kappa,rho,sigma,mu,T);
    cf(1)=0.5*cf(1);

    %Option price  
    Re=real(repmat(cf,1,length(x)).*exp(1i*omega*(x'-a))) ;  
    Vk=2/(b-a)*(chi(0,b,N,a,b)'-psi(0,b,N,a,b)'); 
    ret=K.*exp(-r*T).*(Re'*Vk);
end

function chivector=chi(x1, x2, N, a, b)
    k=0:1:N-1;
    temp=k*pi/(b-a);
    chivector=(1./(1+temp.^2)).*(cos(temp*(x2-a))*exp(x2)-cos(temp*(x1-a))*exp(x1)+temp.*sin(temp*(x2-a))*exp(x2)-temp.*sin(temp*(x1-a))*exp(x1));
end

function psivector=psi(x1, x2, N, a, b)
    k=1:N-1;
    temp=k*pi/(b-a);
    psitemp=(1./(temp)).*(sin(temp*(x2-a))-sin(temp*(x1-a)));
    psivector=[x2-x1 psitemp];
end


function phi=ChFHeston(omega,u0,ubar,lambda,rho,eta,mu,dt)

    D=sqrt((lambda-i*rho*eta*omega).^2+(omega.^2+i*omega)*eta^2);
    G=(lambda-i*rho*eta*omega-D)./(lambda-i*rho*eta*omega+D);
    phi=exp(i*omega*mu*dt+u0/eta^2*(1-exp(-D*dt))./(1-G.*exp(-D*dt))...
    .*(lambda-i*rho*eta*omega-D))...
    .*exp(lambda*ubar/eta^2*(dt*(lambda-i*rho*eta*omega-D)-2*log((1-G.*exp(-D*dt))./(1-G))));
end
