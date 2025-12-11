function Twc = tumPoseToTransform(txyz, quat)
% Convert TUM RGB-D format pose to camera-to-world transformation matrix
% Reference: TUM RGB-D dataset format standard

if numel(txyz) ~= 3
    error('txyz must be a 3-element vector [tx, ty, tz]');
end
if numel(quat) ~= 4
    error('quat must be a 4-element vector [qx, qy, qz, qw]');
end

quat = quat(:) / norm(quat);

qx = quat(1);
qy = quat(2);
qz = quat(3);
qw = quat(4);

R = [1 - 2*qy^2 - 2*qz^2,   2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw;
     2*qx*qy + 2*qz*qw,   1 - 2*qx^2 - 2*qz^2,   2*qy*qz - 2*qx*qw;
     2*qx*qz - 2*qy*qw,   2*qy*qz + 2*qx*qw,   1 - 2*qx^2 - 2*qy^2];

t = txyz(:);

Twc = [R, t;
       0, 0, 0, 1];

end
