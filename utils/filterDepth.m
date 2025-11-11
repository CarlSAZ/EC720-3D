function depthFiltered = filterDepth(depth, varargin)
%FILTERDEPTH Apply basic filtering to a raw depth map.
%   depthFiltered = FILTERDEPTH(depth, ...) removes invalid depth values
%   and optionally clips or smooths the depth map. Name-value pairs:
%       'minDepth'     - minimum valid depth in metres (default: 0)
%       'maxDepth'     - maximum valid depth in metres (default: inf)
%       'medianKernel' - size of the median filter window (default: 0, skip)
%
%   Depth values equal to zero or NaN remain zero in the output.

narginchk(1, inf);
validateattributes(depth, {'double','single'}, {'2d'}, mfilename, 'depth', 1);

if ~isa(depth, 'double')
    depth = double(depth);
end

p = inputParser;
addParameter(p, 'minDepth', 0);
addParameter(p, 'maxDepth', inf);
addParameter(p, 'medianKernel', 0);
parse(p, varargin{:});
opts = p.Results;

depthFiltered = depth;
invalidMask = isnan(depthFiltered) | depthFiltered <= 0;

if ~isinf(opts.maxDepth)
    invalidMask = invalidMask | depthFiltered > opts.maxDepth;
end
if opts.minDepth > 0
    invalidMask = invalidMask | depthFiltered < opts.minDepth;
end

depthFiltered(invalidMask) = 0;

if opts.medianKernel >= 3
    k = opts.medianKernel;
    if mod(k,2) == 0
        k = k + 1; % enforce odd window size
    end
    depthFiltered = medfilt2(depthFiltered, [k, k], 'symmetric');
    depthFiltered(invalidMask) = 0; % ensure invalid stays zero after filtering
end

end

