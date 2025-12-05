function GlobalXYZ = bronnXYZtoGlobal(XYZ,pose)
%BRONNXYZTOGLOBAL Summary of this function goes here
%   Detailed explanation goes here

% this is it's own inverse
Tros = [
    -1 0 0 0
    0 0 1 0
    0 1 0 0
    0 0 0 1];

Tm = [
    1.0157    0.1828   -0.2389    0.0113
    0.0009   -0.8431   -0.6413   -0.0098
    -0.3009   0.6147   -0.8085    0.0111
    0         0         0         1.0000];

quat = quaternion(pose.quat);
t = pose.txyz.';
R = quat.rotmat("frame");
T0 = [[R,t];[0 0 0 1]];

Tg = Tros*T0*Tros*Tm;


end

