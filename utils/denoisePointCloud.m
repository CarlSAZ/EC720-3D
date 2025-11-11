function [XYZclean, RGBclean, keepMask] = denoisePointCloud(XYZ, RGB, varargin)
%DENOISEPOINTCLOUD Remove isolated points from a point cloud.
%   [XYZclean, RGBclean, keepMask] = DENOISEPOINTCLOUD(XYZ, RGB, ...)
%   keeps points that have at least minNeighbors within the specified
%   radius using rangesearch. Inputs XYZ/RGB are 3xN arrays.
%
%   Name-value pairs:
%       'radius'        - neighbourhood radius (metres, default 0.05)
%       'minNeighbors'  - minimum number of neighbours (including itself)
%                         required to keep a point (default 5)

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

