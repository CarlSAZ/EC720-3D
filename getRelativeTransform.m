function [RelRot,RelTrans] = getRelativeTransform(Rt_1,Rt_2)

R2inv = inv(Rt_2(1:3,1:3));

RelRot = R2inv*Rt_1(1:3,1:3);
RelTrans = R2inv*(Rt_1(1:3,4) - Rt_2(1:3,4));