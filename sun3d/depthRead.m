% Source: SUN3D Database
% https://sun3d.cs.princeton.edu/toolbox/
function depth = depthRead(filename)
    depth = imread(filename);
    depth = bitor(bitshift(depth,-3), bitshift(depth,16-3));
    depth = single(depth)/1000;
end