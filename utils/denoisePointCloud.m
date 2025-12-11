function [XYZclean, RGBclean, keepMask] = denoisePointCloud(XYZ, RGB, varargin)
% Remove isolated points from point cloud using rangesearch

narginchk(2, inf);
validateattributes(XYZ, {'double'}, {'size',[3 NaN]}, mfilename, 'XYZ', 1);
validateattributes(RGB, {'double'}, {'size',[3 size(XYZ,2)]}, mfilename, 'RGB', 2);

p = inputParser;
addParameter(p, 'radius', 0.05);
addParameter(p, 'minNeighbors', 5);
parse(p, varargin{:});
opts = p.Results;

if isempty(XYZ)
    XYZclean = XYZ;
    RGBclean = RGB;
    keepMask = false(1, 0);
    return;
end

points = XYZ';
[neighbors, ~] = rangesearch(points, points, opts.radius);

numNeighbors = cellfun(@numel, neighbors);
keepMask = numNeighbors >= opts.minNeighbors;

XYZclean = XYZ(:, keepMask);
RGBclean = RGB(:, keepMask);

end

