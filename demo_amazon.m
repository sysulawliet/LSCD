clc;
clear;
load amazon_data.mat

k = 50;
gamma = 0.01;
iteration = 500;
ku = 0.1;
kh = 0.1;

[mae,rmse,U,H,V] = LSCD(train_domain,test_domain,k,gamma,iteration,ku,kh);

save amazon_res.mat

