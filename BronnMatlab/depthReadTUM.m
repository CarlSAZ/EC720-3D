function depth_m = depthReadTUM(filename)
    depth_m = imread(filename);
    depth_m = double(depth_m)/5000;
end