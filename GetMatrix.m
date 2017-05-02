function R = GetMatrix(data,m)
n = max(data(:,2));
R = zeros(m,n);

for i = 1:size(data,1)
    R(data(i,1),data(i,2)) = data(i,3);
end

end

