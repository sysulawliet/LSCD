function error = MAE(predict_scores,target_result)
format short
[m,n] = size(predict_scores);
N = m*n;
error = sum(sum(abs(predict_scores - target_result)))/N;
end

