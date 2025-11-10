function [delta,xp,yp] = affineMotion(R,T,f,x,y,Z)

Z(Z==0) = inf;

denom = R(3,1).*x + R(3,2).*y + R(3,3).*f + f.*T(3)./Z;
xp = f*(R(1,1).*x + R(1,2).*y + R(1,3).*f + f.*T(1)./Z)./denom;
yp = f*(R(2,1).*x + R(2,2).*y + R(2,3).*f + f.*T(1)./Z)./denom;

delta = zeros([2 size(x)]);
delta(1,:) = x(:) - xp(:);
delta(2,:) = y(:) - yp(:);