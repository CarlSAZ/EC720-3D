function XYZcamera = depth2XYZcameraTUM(fx,fy,cx,cy, depth)
    [x,y] = meshgrid(1:640, 1:480);
    XYZcamera(:,:,1) = (x-cx).*depth/fx;
    XYZcamera(:,:,2) = (y-cy).*depth/fy;
    XYZcamera(:,:,3) = depth;
    XYZcamera(:,:,4) = depth~=0;
end