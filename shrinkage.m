function Y = shrinkage( X,e )
I1 = (X > e);
I2 = (X < -e);
Y = (X - e).*I1 + (X + e).*I2;
end

