function depthFiltered = filterDepth(depth, varargin)
% Apply basic filtering to raw depth map

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
        k = k + 1;
    end
    depthFiltered = medfilt2(depthFiltered, [k, k], 'symmetric');
    depthFiltered(invalidMask) = 0;
end

end

