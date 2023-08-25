
%% Variables

n = 20; m = 1; ndim = 1;

sign = 0.2; sigf = 0.2;
a = rand(ndim,1); b = rand(1);
X = rand(n,ndim); X_ = rand(m,ndim);
M = eye(ndim);

w = rand(ndim, 1); d = rand(1);
f = generate_ytrue(X,w,d);
f_ = generate_ytrue(X_,w,d);

Kxx = rand(n,n);
Kxx_ = rand(n,m);
Kx_x = rand(m,n);
Kx_x_ = rand(m,m);

%% Quick gradient descent-based initialization

nepoch_pre = 200;
for i = 1:nepoch_pre
    mx = meanfunc(X,a,b);
    C = mx - f;
    dcda = X' * C;
    dcdb = mean(C);
    a = a - dcda * 0.004;
    b = b - dcdb * 0.004;
    if i > nepoch_pre - 4
        disp(mean(C));
    end
end

%% Train routine

nepochs = 20;
nbatches = 100;
for epoch = 1:nepochs
    for batch = 1:nbatches
        X = rand(n,ndim); f = generate_ytrue(X,w,d);
        Kxx = get_k_mat(sigf, X,X, n,n, M);
        

         %% FOWARD PROP
%         for testpnt = 1:n
%             X_ = X(testpnt,:);
%             Kxx_ = get_k_mat(sigf, X,X_, n,m, M);
%             Kx_x = get_k_mat(sigf, X_,X, m,n, M);
%             Kx_x_ = get_k_mat(sigf, X_,X_, m,m, M);
%         
%             mx = meanfunc(X,a,b);
%             mx_ = meanfunc(X_,a,b);
%         
%             MID = inv(Kxx + sign^2 * eye(n));
%             meu_ = mx_ + Kx_x * MID * (f - mx);
%             sig2_ = Kx_x_ - Kx_x * MID * Kxx_;
%         end

        %% BACK PROP
        % minimize: C = log( | 2pi * (Kxx + sign^2 * I)) | ) + \
        %           (f-m(x))^T * (Kxx + sign^2 * I)^(-1) * (f-m(x))
        % 
        %           => C = log( | B | ) + (f_mx)^T * A * (f_mx)
    
        % general vars
        f_mx = f-mx;
        A = Kxx + sign^2 * eye(n);

        A = A + 0.00003 * eye(n);      % avoid singularity

        B = 2*pi*A;
        
        dcdB = ( 1 / (det(B)*log(10)) ) * cofac(B);
        dcdA = f_mx * f_mx' * (-1) * A * A;
    
        % dcd_sign
        dcd_sign = trace(dcdB * 4*pi*sign*eye(n)) / n;       % dcd_B
        dcd_sign = dcd_sign + trace(dcdA * 2*sign*eye(n)) / n;  % dcdA
    
        % dcd_sigf
        dcd_Kxx = 2*pi*dcdB;
        dcd_sigf = mean( (2*sigf / sigf^2) .* Kxx .* dcd_Kxx, "all" );
        dcd_sigf = dcd_sigf + mean( (2*sigf / sigf^2) .* Kxx .* dcdA, "all" );
        
        % dcdM
        dcdM = zeros(ndim);
%         for i = 1:n
%             for j = 1:n
%                 s = sqrt(3) * norm(M * (X(i,:) - X(j,:))');
%                 dcds = dcd_Kxx(i,j) * sigf^2 * (-s * exp(-s));
%                 dsdM = sqrt(3) * diag(2*M*x .* x);
%                 dcdM = dcdM + dcds * dsdM * (1/n^2);
%             end
%         end

%         % display
%         if epoch > nepochs - 10
%             disp(sigf + "   " + sign + "     " + sig2_);
%         end
        
        % dcd_mx, dcda, dcdb
        dcd_mx = -A*f;
        for j = 1:n
            dcd_mx(j,:) = dcd_mx(j,:) - sum(A(j,:)) * f(n);
            dcd_mx(j,:) = dcd_mx(j,:) - 2 * sum(A(j,:)) * mx(j);
        end
        dcdb = mean(dcd_mx);
        dcda = X' * dcd_mx;
 
        % dcdM
        dcdM = zeros(ndim);
        for i = 1:n
            for j = 1:n
                s = norm( M * (X(i,:) - X(j,:))' )^2;
                dcds = dcd_Kxx(i,j) * sigf^2 * exp(-sqrt(3*s)) * (-0.5)*(3*s)^(-0.5)*3 + sigf^2 *0.5*(3*s)^(-0.5)*exp(-sqrt(3*s))*3 + sqrt(3*s)*exp(-sqrt(3*s))*(0.5)*(3*s)^(-0.5);
                dsdM = diag(2 * M * (X(i,:) - X(j,:))' .* (X(i,:) - X(j,:))');
                dcdM = dcds * dsdM;
            end
        end
        
        % adjustments
        lrate = 0.0005;
        sign = sign - dcd_sign * lrate;
        sigf = sigf - dcd_sigf * lrate;
        a = a - dcda * lrate;
        b = b - dcdb * lrate;
%         M = M - dcdM * lrate;
    end

end


%% TEST ROUTINE

n = 140;
m = 1;

% X = sort(X);
X = sort( rand(n,ndim).*1 ); X_ = rand(m,ndim);
ypred = rand(n,1);
ytrue = generate_ytrue(X,w,d);
devs = rand(n,1);
devs2 = rand(n,1);

Kxx_ = rand(n,m);
Kx_x = rand(m,n);
Kxx = rand(n,n);
Kx_x_ = rand(m,m);

for i = 1:n
        X_ = X(i,:);
        Kxx = get_k_mat(sigf, X,X, n,n, M);
        Kxx_ = get_k_mat(sigf, X,X_, n,m, M);
        Kx_x = get_k_mat(sigf, X_,X, m,n, M);
        Kx_x_ = get_k_mat(sigf, X_,X_, m,m, M);
    
        mx = meanfunc(X,a,b);
        mx_ = meanfunc(X_,a,b);
    
        MID = inv(Kxx + sign^2 * eye(n));
        
        MID = MID + 0.000003 * eye(n);      % avoid singularity

        meu_ = mx_ + Kx_x * MID * (ytrue - mx);
        sig2_ = Kx_x_ - Kx_x * MID * Kxx_;

        devs2(i) = Kx_x_ + sign^2; % sqrt(Kxx(i,i));   % mean(Kxx(i,:)) / Kxx(i,i);   % sqrt(sig2_);        % Kx_x_;      % sqrt(Kxx(i,i));     % sig2_;
%         devs(i) = 2 * sqrt(sig2_);
        ypred(i) = meu_;
end

plot(X(:,1),ytrue,'g','linestyle','--'); hold on;
plot(X(:,1),ypred,'r','linestyle',':'); hold on;
% 
plot(X(:,1),ypred+devs2,'r','linestyle',':'); hold on;
plot(X(:,1),ypred-devs2,'r','linestyle',':'); hold on;
% 
% plot(X(:,1),ypred+devs,'o','linestyle','-.'); hold on;
% plot(X(:,1),ypred-devs,'o','linestyle','-.'); hold on;

% disp(devs);
disp(devs2);
hold off;



%% Helper functions

function y = generate_ytrue(x,w,d)
    nrows = size(x,1);
    % noise = (rand(nrows,1)-0.5) / 5; %./ max(1,rand(1)*10);
    noise = randn(nrows,1) / 90;
    y = ones(nrows,1);

    bias = zeros(nrows,1);
%     bias = (5 .* noise .* ( x(:,1)>0.5 ) );
    
    if size(d,1) == 1
        y = x * w + repmat(d,nrows,1) + noise + bias;
    else
        y = x * w + d + noise + bias;
    end
end

function kyz = get_k_mat(sigf,Y,Z, n, m, M)
    kyz = ones(n,m);
    for i =1:n
        for j = 1:m
            kyz(i,j) = get_k_idx(sigf,Y(i,:)',Z(j,:)', M);
        end
    end
end


function k = get_k_idx(sigf, x, x_, M)
    s = sqrt(3 * norm(M * (x-x_))^2);
    k = sigf^2 * (1 + s) * exp(-s);
end

function f=meanfunc(x,a,b)
    f = x * a + repmat(b,size(x,1),1);
end

function res = cofac(A)
    res = (det(A) * inv(A))';
end
