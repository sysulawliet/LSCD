function error = RMSE( predict_scores,target_result )
format short
[m,n] = size(predict_scores);
N = m*n;
error = sqrt(sum(sum((predict_scores - target_result).^2))/N);

end

