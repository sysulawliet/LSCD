function [mae,rmse,U,H,V] = LSCD(train_domain,test_domain,k,gamma,iteration,ku,kh)
rmin = 0;
rmax = 5;

kv = 0.1;

T = size(train_domain,2) - 1;
ms = zeros(1,T);

for t = 1:T
    ms(t) = max(train_domain{t}(:,1));
    tmp = max(test_domain{t}(:,1));
    if ms(t) < tmp
        ms(t) = tmp;
    end
end
m = max(ms);

U = 0.1*rand(m,k)./sqrt(k);
mean_rating = zeros(1,T);
ns = zeros(1,T);
R = cell(1,T);
I = cell(1,T);
H = cell(1,T);
V = cell(1,T);
D = cell(1,T);
P = cell(1,T);


mae = zeros(T + 1,iteration);
rmse = zeros(T + 1,iteration);

for t = 1:T
    R{t} = GetMatrix(train_domain{t},m);
    [~,ns(t)] = size(R{t});
    tmp = R{t};
    tmp(tmp > 0) = 1;
    I{t} = tmp;
    mean_rating(t) = sum(sum(R{t}))/sum(sum(I{t}));
    H{t} = 0.1*rand(m,k)./sqrt(k);
    V{t} = 0.1*rand(ns(t),k)./sqrt(k);
    P{t} = (U + H{t})*V{t}' + mean_rating(t);
    P{t}(P{t} > rmax) = rmax;
    P{t}(P{t} < rmin) = rmin;
end

for l = 1:iteration
    %update U
    dU = zeros(size(U));
    for t = 1:T
        D{t} = (P{t} - R{t}).*I{t};
        dU = dU + D{t}*V{t};
    end
    Z = U - gamma.*dU;
    Z(isnan(Z)) = 0;
    [Q,S,W] = svd(Z);
    U = Q*shrinkage(S,gamma*ku)*W';
    
    %update H
    for t = 1:T
        P{t} = (U + H{t})*V{t}' + mean_rating(t);
        D{t} = (P{t} - R{t}).*I{t};
        H{t} =shrinkage( H{t} - gamma.*D{t}*V{t},gamma*kh);
    end
    
    %update V
    for t = 1:T
        P{t} = (U + H{t})*V{t}' + mean_rating(t);
        D{t} = (P{t} - R{t}).*I{t};
        V{t} = V{t} - gamma.*(D{t}'*(U + H{t}) + kv.*V{t});
    end
    
    all_pr = [];
    all_rr = [];  
    for t = 1:T
        P{t} = (U + H{t})*V{t}' + mean_rating(t);
        P{t}(P{t} > rmax) = rmax;
        P{t}(P{t} < rmin) = rmin;
        rr = test_domain{t}(:,3);
        test_num = size(test_domain{t},1);
        pr = zeros(test_num,1);
        for i = 1:test_num
            uid = test_domain{t}(i,1);
            iid = test_domain{t}(i,2);
            pr(i) = P{t}(uid,iid);
        end
        all_pr = [all_pr;pr];
        all_rr = [all_rr;rr];
        mae(t,l) = MAE(pr,rr);
        rmse(t,l) = RMSE(pr,rr);
        fprintf('LSCD: Iteration: l = %d, domain: d = %d, MAE = %f, RMSE = %f\n',l,t,mae(t,l),rmse(t,l));
    end
    mae(T + 1,l) = MAE(all_pr,all_rr);
    rmse(T + 1,l) = RMSE(all_pr,all_rr);
    fprintf('LSCD: Iteration: l = %d, domain: all = %d, MAE = %f, RMSE = %f\n',l,t,mae(t,l),rmse(t,l));
end

end

