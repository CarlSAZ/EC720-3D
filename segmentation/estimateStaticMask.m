function result = estimateStaticMask(pointsCam, coloursCam, mapPrediction, varargin)
%ESTIMATESTATICMASK Segment static and dynamic observations via energy terms.
%   result = ESTIMATESTATICMASK(pointsCam, coloursCam, mapPrediction) compares
%   the current camera-frame observations against a predicted static map (from
%   PREDICTSTATICMAP) to compute per-point energies and a static mask.
%
%   Inputs:
%       pointsCam      - 3xN coordinates in the current camera frame (metres)
%       coloursCam     - 3xN RGB values in [0, 1]
%       mapPrediction  - struct with fields XYZcam, normalsCam, colours,
%                        confidence (as returned by PREDICTSTATICMAP)
%
%   Name-value pairs:
%       'NormalsCam'          - 3xN observed normals (optional)
%       'GeometryThreshold'   - point-to-plane threshold (default 0.04 m)
%       'ColorThreshold'      - colour difference threshold (default 0.15)
%       'NormalThreshold'     - angle threshold in radians (default pi/4)
%       'EnergyWeights'       - struct with fields geometry, colour,
%                               normal, prior (defaults [1,1,0.5,1])
%       'EnergyThreshold'     - classify as static if energy <= threshold (1.5)
%       'PriorDynamicMask'    - logical 1xN prior dynamic flag (optional)
%       'ConfidenceWeighting' - logical flag (default true)
%       'MinConfidence'       - lower bound when normalising confidence (0.1)
%
%   Output struct fields:
%       staticMask       - logical 1xN static labels
%       dynamicMask      - logical 1xN dynamic labels
%       energy           - 1xN combined energy values
%       geoResidual      - 1xN point-to-plane residuals
%       colourResidual   - 1xN colour residuals
%       normalResidual   - 1xN normal angle residuals
%       weights          - 1xN confidence weights applied
%       correspondences  - indices into the predicted map

narginchk(3, inf);
validateattributes(pointsCam, {'double'}, {'size',[3, NaN]}, mfilename, 'pointsCam', 1);
validateattributes(coloursCam, {'double'}, {'size',[3, size(pointsCam, 2)]}, mfilename, 'coloursCam', 2);
validateMapPrediction(mapPrediction);

parser = inputParser;
addParameter(parser, 'NormalsCam', [], @(x) isempty(x) || ...
    (isfloat(x) && size(x,1)==3 && size(x,2)==size(pointsCam,2)));
addParameter(parser, 'GeometryThreshold', 0.04, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'ColorThreshold', 0.15, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'NormalThreshold', pi/4, @(x) validateattributes(x, {'double'}, {'scalar','positive','<=',pi}));
addParameter(parser, 'EnergyWeights', struct(), @(x) isempty(x) || isstruct(x));
addParameter(parser, 'EnergyThreshold', 1.5, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'PriorDynamicMask', false(1, size(pointsCam,2)), @(x) islogical(x) && isvector(x) && numel(x)==size(pointsCam,2));
addParameter(parser, 'ConfidenceWeighting', true, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'MinConfidence', 0.1, @(x) validateattributes(x, {'double'}, {'scalar','nonnegative'}));
parse(parser, varargin{:});
opts = parser.Results;

weightsCfg = struct('geometry', 1.0, 'colour', 1.0, 'normal', 0.5, 'prior', 1.0);
if ~isempty(opts.EnergyWeights)
    fields = fieldnames(opts.EnergyWeights);
    for idx = 1:numel(fields)
        weightsCfg.(fields{idx}) = opts.EnergyWeights.(fields{idx});
    end
end

numPoints = size(pointsCam, 2);

result = struct();
result.staticMask = false(1, numPoints);
result.dynamicMask = true(1, numPoints);
result.energy = nan(1, numPoints);
result.geoResidual = nan(1, numPoints);
result.colourResidual = nan(1, numPoints);
result.normalResidual = nan(1, numPoints);
result.weights = zeros(1, numPoints);
result.correspondences = zeros(1, numPoints);

if numPoints == 0
    return;
end

if isempty(mapPrediction.XYZcam)
    warning('estimateStaticMask:EmptyMap', ...
        'Predicted static map is empty; marking all points as dynamic.');
    return;
end

% Validate inputs before knnsearch
if any(isnan(mapPrediction.XYZcam(:))) || any(isinf(mapPrediction.XYZcam(:)))
    error('estimateStaticMask:InvalidMap', 'Map prediction contains NaN or Inf values.');
end
if any(isnan(pointsCam(:))) || any(isinf(pointsCam(:)))
    error('estimateStaticMask:InvalidPoints', 'Input points contain NaN or Inf values.');
end

try
    [indices, distances] = knnsearch(mapPrediction.XYZcam', pointsCam');
catch ME
    error('estimateStaticMask:KnnSearchFailed', ...
        'knnsearch failed: %s. Map size: %dx%d, Points size: %dx%d', ...
        ME.message, size(mapPrediction.XYZcam, 1), size(mapPrediction.XYZcam, 2), ...
        size(pointsCam, 1), size(pointsCam, 2));
end

% Validate indices
if any(indices < 1) || any(indices > size(mapPrediction.XYZcam, 2))
    error('estimateStaticMask:InvalidIndices', ...
        'KNN search returned invalid indices. Range: [%d, %d], Map size: %d', ...
        min(indices), max(indices), size(mapPrediction.XYZcam, 2));
end

correspondenceNormals = mapPrediction.normalsCam(:, indices);
correspondenceColours = mapPrediction.colours(:, indices);
correspondenceConfidence = mapPrediction.confidence(indices);

delta = pointsCam - mapPrediction.XYZcam(:, indices);
geoResidual = abs(sum(correspondenceNormals .* delta, 1));

colourResidual = vecnorm(coloursCam - correspondenceColours, 2, 1);

if ~isempty(opts.NormalsCam)
    normalDots = sum(opts.NormalsCam .* correspondenceNormals, 1);
    normalDots = max(-1, min(1, normalDots));
    normalResidual = acos(normalDots);
else
    normalResidual = zeros(1, numPoints);
end

priorEnergy = zeros(1, numPoints);
priorEnergy(opts.PriorDynamicMask) = weightsCfg.prior;

% Normalised residuals.
geoNorm = geoResidual / opts.GeometryThreshold;
colourNorm = colourResidual / opts.ColorThreshold;
if ~isempty(opts.NormalsCam)
    normalNorm = normalResidual / opts.NormalThreshold;
else
    normalNorm = zeros(1, numPoints);
end

rawEnergy = weightsCfg.geometry * geoNorm.^2 + ...
    weightsCfg.colour * colourNorm.^2 + ...
    weightsCfg.normal * normalNorm.^2 + priorEnergy;

if opts.ConfidenceWeighting
    confidence = max(opts.MinConfidence, correspondenceConfidence);
    maxConf = max(confidence);
    if maxConf > 0
        weights = confidence / maxConf;
    else
        weights = ones(1, numPoints);
    end
else
    weights = ones(1, numPoints);
end

% Combine raw energy with confidence weights. We want lower-confidence
% correspondences to contribute less, so multiply instead of divide.
energy = rawEnergy .* weights;

% Boolean gating based on individual residual thresholds
geoMask = geoResidual <= opts.GeometryThreshold;
colourMask = colourResidual <= opts.ColorThreshold;
if ~isempty(opts.NormalsCam)
    normalMask = normalResidual <= opts.NormalThreshold;
else
    normalMask = true(1, numPoints);
end

baseStaticMask = geoMask & colourMask & normalMask;
energyStaticMask = energy <= opts.EnergyThreshold;

% Combine gating with energy-based decision
staticMask = baseStaticMask | (energyStaticMask & geoMask);
dynamicMask = ~staticMask;

result.staticMask = staticMask;
result.dynamicMask = dynamicMask;
result.energy = energy;
result.geoResidual = geoResidual;
result.colourResidual = colourResidual;
result.normalResidual = normalResidual;
result.weights = weights;
result.correspondences = indices;
result.distances = distances;
end

function validateMapPrediction(prediction)
requiredFields = {'XYZcam', 'normalsCam', 'colours', 'confidence'};
for f = requiredFields
    if ~isfield(prediction, f{1})
        error('estimateStaticMask:MissingField', ...
            'Map prediction missing field ''%s''.', f{1});
    end
end
end

