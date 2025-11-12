function map = initializeStaticMap(pointCloud, normals, varargin)
%INITIALIZESTATICMAP Create a surfel-based static background map.
%   map = INITIALIZESTATICMAP(pointCloud, normals) instantiates a static
%   background map using the supplied world-frame point cloud structure.
%
%   Inputs:
%       pointCloud.XYZ  - 3xN world coordinates (metres)
%       pointCloud.RGB  - 3xN colours in [0, 1]
%       normals         - 3xN surfel normals (world frame)
%
%   Name-value pairs:
%       'DefaultConfidence'  - initial confidence per surfel (default: 1)
%       'DefaultRadius'      - nominal surfel radius in metres (default: 0.03)
%       'MergeRadius'        - association radius when fusing (default: 0.05)
%       'ObservationWeight'  - incremental observation weight (default: 1)
%       'MaxConfidence'      - cap for surfel confidence (default: 10)
%       'MinConfidence'      - minimum confidence value (default: 0.1)
%       'ConfidenceDecay'    - per-update decay factor (default: 0)
%
%   Output:
%       map struct with fields positions, normals, colours, confidence,
%       radius, and params for subsequent updates.

narginchk(2, inf);
validateattributes(pointCloud, {'struct'}, {'scalar'}, mfilename, 'pointCloud', 1);
validateattributes(normals, {'double'}, {'size',[3, NaN]}, mfilename, 'normals', 2);

requiredFields = {'XYZ', 'RGB'};
for f = requiredFields
    if ~isfield(pointCloud, f{1})
        error('initializeStaticMap:MissingField', ...
            'Point cloud struct must contain field ''%s''.', f{1});
    end
end

XYZ = pointCloud.XYZ;
RGB = pointCloud.RGB;

if size(XYZ, 1) ~= 3
    error('initializeStaticMap:InvalidXYZ', 'XYZ must be 3xN.');
end
if size(RGB, 1) ~= 3 || size(RGB, 2) ~= size(XYZ, 2)
    error('initializeStaticMap:InvalidRGB', 'RGB must be 3xN matching XYZ.');
end
if size(normals, 2) ~= size(XYZ, 2)
    error('initializeStaticMap:NormalCount', ...
        'Normals must have the same number of columns as XYZ.');
end

parser = inputParser;
addParameter(parser, 'DefaultConfidence', 1.0, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'DefaultRadius', 0.03, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'MergeRadius', 0.05, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'ObservationWeight', 1.0, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'MaxConfidence', 10.0, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'MinConfidence', 0.1, @(x) validateattributes(x, {'double'}, {'scalar','nonnegative'}));
addParameter(parser, 'ConfidenceDecay', 0.0, @(x) validateattributes(x, {'double'}, {'scalar','>=',0,'<',1}));
parse(parser, varargin{:});
opts = parser.Results;

numSurfels = size(XYZ, 2);

map = struct();
map.positions = XYZ;
map.normals = normalizeNormals(normals);
map.colours = RGB;
map.confidence = ones(1, numSurfels) * opts.DefaultConfidence;
map.radius = ones(1, numSurfels) * opts.DefaultRadius;
map.params = struct( ...
    'mergeRadius', opts.MergeRadius, ...
    'observationWeight', opts.ObservationWeight, ...
    'defaultRadius', opts.DefaultRadius, ...
    'maxConfidence', opts.MaxConfidence, ...
    'minConfidence', opts.MinConfidence, ...
    'confidenceDecay', opts.ConfidenceDecay);
end

%% Local helper
function normalsOut = normalizeNormals(normalsIn)
normalsOut = zeros(size(normalsIn));
if isempty(normalsIn)
    return;
end
norms = sqrt(sum(normalsIn.^2, 1));
zeroMask = norms < eps;
normalsOut(:, ~zeroMask) = normalsIn(:, ~zeroMask) ./ norms(~zeroMask);
end

