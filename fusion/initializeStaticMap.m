function map = initializeStaticMap(pointCloud, normals, varargin)
% Create surfel-based static background map

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

function normalsOut = normalizeNormals(normalsIn)
normalsOut = zeros(size(normalsIn));
if isempty(normalsIn)
    return;
end
norms = sqrt(sum(normalsIn.^2, 1));
zeroMask = norms < eps;
normalsOut(:, ~zeroMask) = normalsIn(:, ~zeroMask) ./ norms(~zeroMask);
end

