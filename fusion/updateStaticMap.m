function map = updateStaticMap(map, Twc, staticPointsCam, staticNormalsCam, staticColours, varargin)
%UPDATESTATICMAP Fuse observed static points into the background map.
%   map = UPDATESTATICMAP(map, Twc, staticPointsCam, staticNormalsCam, staticColours)
%   integrates camera-frame static observations into the world-frame map.
%
%   Inputs:
%       map                - background map struct from INITIALIZESTATICMAP
%       Twc                - 4x4 camera-to-world transform for current frame
%       staticPointsCam    - 3xM points in camera coordinates (metres)
%       staticNormalsCam   - 3xM normals in camera coordinates
%       staticColours      - 3xM colours in [0, 1]
%
%   Name-value pairs:
%       'MergeRadius'        - override map.params.mergeRadius
%       'ObservationWeight'  - override map.params.observationWeight
%       'MaxConfidence'      - override map.params.maxConfidence
%       'MinConfidence'      - override map.params.minConfidence
%       'ConfidenceDecay'    - override map.params.confidenceDecay
%       'DefaultRadius'      - override map.params.defaultRadius

narginchk(5, inf);
validateStaticMap(map);
validateattributes(Twc, {'double'}, {'size',[4,4]}, mfilename, 'Twc', 2);
validateattributes(staticPointsCam, {'double'}, {'size',[3, NaN]}, mfilename, 'staticPointsCam', 3);
validateattributes(staticNormalsCam, {'double'}, {'size',[3, size(staticPointsCam,2)]}, mfilename, 'staticNormalsCam', 4);
validateattributes(staticColours, {'double'}, {'size',[3, size(staticPointsCam,2)]}, mfilename, 'staticColours', 5);

parser = inputParser;
addParameter(parser, 'MergeRadius', map.params.mergeRadius, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'ObservationWeight', map.params.observationWeight, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'MaxConfidence', map.params.maxConfidence, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'MinConfidence', map.params.minConfidence, @(x) validateattributes(x, {'double'}, {'scalar','nonnegative'}));
addParameter(parser, 'ConfidenceDecay', map.params.confidenceDecay, @(x) validateattributes(x, {'double'}, {'scalar','>=',0,'<',1}));
addParameter(parser, 'DefaultRadius', map.params.defaultRadius, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
parse(parser, varargin{:});
opts = parser.Results;

% Apply global confidence decay before integrating new data.
if opts.ConfidenceDecay > 0 && ~isempty(map.confidence)
    map.confidence = max(opts.MinConfidence, map.confidence * (1 - opts.ConfidenceDecay));
end

% Transform observations into world frame.
Rw = Twc(1:3, 1:3);
tw = Twc(1:3, 4);
pointsWorld = Rw * staticPointsCam + tw;
normalsWorld = normalizeNormals(Rw * staticNormalsCam);

numObservations = size(pointsWorld, 2);

if numObservations == 0
    return;
end

if isempty(map.positions)
    map.positions = pointsWorld;
    map.normals = normalsWorld;
    map.colours = staticColours;
    map.confidence = ones(1, numObservations) * opts.ObservationWeight;
    map.radius = ones(1, numObservations) * opts.DefaultRadius;
    return;
end

% Validate inputs before knnsearch
if any(isnan(map.positions(:))) || any(isinf(map.positions(:)))
    error('updateStaticMap:InvalidMap', 'Map contains NaN or Inf values.');
end
if any(isnan(pointsWorld(:))) || any(isinf(pointsWorld(:)))
    error('updateStaticMap:InvalidPoints', 'Input points contain NaN or Inf values.');
end

try
    [indices, distances] = knnsearch(map.positions', pointsWorld', 'K', 1);
catch ME
    error('updateStaticMap:KnnSearchFailed', ...
        'knnsearch failed: %s. Map size: %dx%d, Points size: %dx%d', ...
        ME.message, size(map.positions, 1), size(map.positions, 2), ...
        size(pointsWorld, 1), size(pointsWorld, 2));
end
matchedMask = distances <= opts.MergeRadius;

obsWeight = opts.ObservationWeight;

% Update existing surfels.
matchedIdx = find(matchedMask);
for k = matchedIdx(:)'
    if k < 1 || k > numel(indices)
        continue;
    end
    surfelIdx = indices(k);
    if surfelIdx < 1 || surfelIdx > size(map.positions, 2)
        continue;
    end
    confSum = map.confidence(surfelIdx) + obsWeight;
    if confSum > 0
        alpha = obsWeight / confSum;
    else
        alpha = 0.5;  % Fallback
    end

    map.positions(:, surfelIdx) = (1 - alpha) * map.positions(:, surfelIdx) + alpha * pointsWorld(:, k);
    map.normals(:, surfelIdx) = normalizeVector((1 - alpha) * map.normals(:, surfelIdx) + alpha * normalsWorld(:, k));
    map.colours(:, surfelIdx) = (1 - alpha) * map.colours(:, surfelIdx) + alpha * staticColours(:, k);
    map.radius(surfelIdx) = max(0.5 * opts.DefaultRadius, ...
        (1 - alpha) * map.radius(surfelIdx) + alpha * opts.DefaultRadius);

    map.confidence(surfelIdx) = min(opts.MaxConfidence, map.confidence(surfelIdx) + obsWeight);
end

% Insert new surfels for unmatched observations.
newMask = ~matchedMask;
numNew = nnz(newMask);
if numNew > 0
    map.positions = [map.positions, pointsWorld(:, newMask)];
    map.normals = [map.normals, normalsWorld(:, newMask)];
    map.colours = [map.colours, staticColours(:, newMask)];
    map.confidence = [map.confidence, ones(1, numNew) * obsWeight];
    map.radius = [map.radius, ones(1, numNew) * opts.DefaultRadius];
end

% Safeguard confidence range.
map.confidence = min(opts.MaxConfidence, max(opts.MinConfidence, map.confidence));
end

%% Local helpers
function validateStaticMap(map)
if ~isstruct(map)
    error('Static map must be a struct.');
end
requiredFields = {'positions', 'normals', 'colours', 'confidence', 'radius', 'params'};
for f = requiredFields
    if ~isfield(map, f{1})
        error('updateStaticMap:MissingField', ...
            'Static map struct missing field ''%s''.', f{1});
    end
end
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

function v = normalizeVector(v)
if all(v == 0)
    return;
end
v = v / norm(v);
end

