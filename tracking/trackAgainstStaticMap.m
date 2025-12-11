% Author: Anqi Wei
% Adapted from StaticFusion
% Reference: StaticFusion: Background Reconstruction for Dense RGB-D SLAM in Dynamic Environments
%            GitHub: https://github.com/raluca-scona/staticfusion

function result = trackAgainstStaticMap(map, initialTwc, pointsCam, mask, varargin)
% Track camera pose using point-to-plane ICP against static map

narginchk(3, inf);
validateStaticMap(map);
validateattributes(initialTwc, {'double'}, {'size',[4,4]}, mfilename, 'initialTwc', 2);
validateattributes(pointsCam, {'double'}, {'size',[3, NaN]}, mfilename, 'pointsCam', 3);

if nargin < 4 || isempty(mask)
    mask = true(1, size(pointsCam, 2));
else
    validateattributes(mask, {'logical'}, {'vector','numel',size(pointsCam,2)}, mfilename, 'mask', 4);
end

parser = inputParser;
addParameter(parser, 'MaxIterations', 10, @(x) validateattributes(x, {'double'}, {'scalar','integer','positive'}));
addParameter(parser, 'MaxCorrespondence', 0.07, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'NormalSimilarity', cosd(45), @(x) validateattributes(x, {'double'}, {'scalar','>=',-1,'<=',1}));
addParameter(parser, 'Damping', 1e-4, @(x) validateattributes(x, {'double'}, {'scalar','nonnegative'}));
addParameter(parser, 'MinInliers', 80, @(x) validateattributes(x, {'double'}, {'scalar','integer','>=',0}));
addParameter(parser, 'Verbose', false, @(x) islogical(x) && isscalar(x));
parse(parser, varargin{:});
opts = parser.Results;

Twc = initialTwc;
success = false;
rmse = NaN;
inlierMask = false(1, sum(mask));
correspondences = zeros(1, sum(mask));
residuals = [];
inliers = false(1, sum(mask));

obsPoints = pointsCam(:, mask);

if isempty(obsPoints)
    warning('trackAgainstStaticMap:NoPoints', 'No observations provided for tracking.');
    result = buildResult(Twc, success, 0, rmse, inlierMask, correspondences, residuals);
    return;
end

mapPositions = map.positions;
mapNormals = map.normals;

if isempty(mapPositions)
    warning('trackAgainstStaticMap:EmptyMap', 'Static map is empty; returning initial pose.');
    result = buildResult(Twc, success, 0, rmse, inlierMask, correspondences, residuals);
    return;
end

iteration = 0;
idx = zeros(1, size(obsPoints, 2));
numObsBase = size(obsPoints, 2);
for iter = 1:opts.MaxIterations
    iteration = iter;
    inliers = false(1, numObsBase);
    
    R = Twc(1:3, 1:3);
    t = Twc(1:3, 4);
    obsWorld = R * obsPoints + t;

    if any(isnan(mapPositions(:))) || any(isinf(mapPositions(:)))
        warning('trackAgainstStaticMap:InvalidMap', 'Map contains NaN or Inf values.');
        break;
    end
    if any(isnan(obsWorld(:))) || any(isinf(obsWorld(:)))
        warning('trackAgainstStaticMap:InvalidObservations', 'Observations contain NaN or Inf values.');
        break;
    end
    
    try
        [idx, dists] = knnsearch(mapPositions', obsWorld');
    catch ME
        warning('trackAgainstStaticMap:KnnSearchFailed', ...
            'knnsearch failed: %s. Map size: %dx%d, Obs size: %dx%d', ...
            ME.message, size(mapPositions, 1), size(mapPositions, 2), ...
            size(obsWorld, 1), size(obsWorld, 2));
        break;
    end
    
    numObsCheck = size(obsWorld, 2);
    if numel(idx) ~= numObsCheck || numel(dists) ~= numObsCheck
        warning('trackAgainstStaticMap:KnnSearchDimensionMismatch', ...
            'knnsearch output size mismatch: idx=%d, dists=%d, expected=%d', ...
            numel(idx), numel(dists), numObsCheck);
        if numel(idx) > numObsCheck
            idx = idx(1:numObsCheck);
            dists = dists(1:numObsCheck);
        elseif numel(idx) < numObsCheck
            warning('trackAgainstStaticMap:UnexpectedShortOutput', ...
                'knnsearch returned fewer results than expected');
            break;
        end
    end
    
    if size(idx, 1) > size(idx, 2)
        idx = idx';
    end
    if size(dists, 1) > size(dists, 2)
        dists = dists';
    end
    
    if any(idx < 1) || any(idx > size(mapPositions, 2))
        warning('trackAgainstStaticMap:InvalidIndices', ...
            'knnsearch returned invalid indices. Range: [%d, %d], Map size: %d', ...
            min(idx), max(idx), size(mapPositions, 2));
        idx(idx < 1) = 1;
        idx(idx > size(mapPositions, 2)) = size(mapPositions, 2);
    end
    
    corresNormals = mapNormals(:, idx);

    numObs = size(obsWorld, 2);
    
    if size(dists, 1) > size(dists, 2)
        dists = dists';
    end
    validDistance = (dists <= opts.MaxCorrespondence);
    if numel(validDistance) ~= numObs
        warning('trackAgainstStaticMap:ValidDistanceSizeMismatch', ...
            'validDistance size (%d) does not match observations (%d)', numel(validDistance), numObs);
        break;
    end
    
    obsNormalsApprox = normalizeColumns(R * obsPoints);
    normalSimilarity = sum(corresNormals .* obsNormalsApprox, 1);
    if size(normalSimilarity, 1) > 1 && size(normalSimilarity, 2) == 1
        normalSimilarity = normalSimilarity';
    end
    validNormals = (normalSimilarity >= opts.NormalSimilarity);
    if numel(validNormals) ~= numObs
        warning('trackAgainstStaticMap:ValidNormalsSizeMismatch', ...
            'validNormals size (%d) does not match observations (%d)', numel(validNormals), numObs);
        break;
    end
    
    if size(validDistance, 1) > 1
        validDistance = validDistance';
    end
    if size(validNormals, 1) > 1
        validNormals = validNormals';
    end
    
    if numel(validDistance) ~= numObs || numel(validNormals) ~= numObs
        warning('trackAgainstStaticMap:DimensionMismatchBeforeAnd', ...
            'Dimension mismatch: validDistance=%d, validNormals=%d, numObs=%d', ...
            numel(validDistance), numel(validNormals), numObs);
        break;
    end
    
    if numel(validDistance) ~= numel(validNormals)
        minLen = min(numel(validDistance), numel(validNormals));
        validDistance = validDistance(1:minLen);
        validNormals = validNormals(1:minLen);
        numObs = minLen;
    end
    
    if ~islogical(validDistance)
        validDistance = logical(validDistance);
    end
    if ~islogical(validNormals)
        validNormals = logical(validNormals);
    end
    
    inliers = validDistance & validNormals;
    
    if ~islogical(inliers)
        warning('trackAgainstStaticMap:InliersNotLogicalAfterAnd', ...
            'inliers is not logical after AND, resetting. Type: %s', class(inliers));
        inliers = false(1, numObs);
    end
    
    if numel(inliers) ~= numObs
        warning('trackAgainstStaticMap:InlierSizeAfterAnd', ...
            'inliers size (%d) != numObs (%d) after AND operation, resetting', numel(inliers), numObs);
        inliers = false(1, numObs);
    end

    if nnz(inliers) < opts.MinInliers
        if opts.Verbose
            fprintf('[trackAgainstStaticMap] Iter %d: insufficient inliers (%d < %d).\n', ...
                iter, nnz(inliers), opts.MinInliers);
        end
        break;
    end

    if numel(inliers) ~= numObs
        warning('trackAgainstStaticMap:InlierSizeMismatch', ...
            'Inliers size (%d) does not match observations (%d)', numel(inliers), numObs);
        break;
    end
    
    if size(idx, 1) > size(idx, 2)
        idx = idx';
    end
    
    validIdxMask = (idx >= 1) & (idx <= size(mapPositions, 2));
    
    if ~islogical(validIdxMask)
        validIdxMask = logical(validIdxMask);
    end
    
    if size(validIdxMask, 1) > size(validIdxMask, 2)
        validIdxMask = validIdxMask';
    end
    
    if numel(validIdxMask) ~= numObs
        warning('trackAgainstStaticMap:ValidIdxSizeMismatch', ...
            'ValidIdxMask size (%d) does not match observations (%d), resetting inliers', numel(validIdxMask), numObs);
        inliers = false(1, numObs);
        break;
    end
    
    if ~islogical(inliers)
        warning('trackAgainstStaticMap:InliersNotLogicalBeforeIdxMask', ...
            'inliers is not logical before idx mask, resetting. Type: %s, size: %d', class(inliers), numel(inliers));
        inliers = false(1, numObs);
    end
    
    if size(inliers, 1) > size(inliers, 2)
        inliers = inliers';
    end
    
    if numel(inliers) ~= numel(validIdxMask)
        warning('trackAgainstStaticMap:DimensionMismatchBeforeAnd', ...
            'Dimension mismatch before AND: inliers=%d, validIdxMask=%d', numel(inliers), numel(validIdxMask));
        inliers = false(1, numObs);
        break;
    end
    
    inliers = inliers & validIdxMask;
    
    if nnz(inliers) < opts.MinInliers
        if opts.Verbose
            fprintf('[trackAgainstStaticMap] Iter %d: insufficient valid inliers (%d < %d).\n', ...
                iter, nnz(inliers), opts.MinInliers);
        end
        break;
    end

    if ~islogical(inliers)
        warning('trackAgainstStaticMap:InliersNotLogical', ...
            'inliers is not logical, resetting. Type: %s, size: %d', class(inliers), numel(inliers));
        inliers = false(1, numObs);
    end
    
    if numel(inliers) ~= numObs
        warning('trackAgainstStaticMap:FinalInlierSizeMismatch', ...
            'Final inliers size (%d) does not match observations (%d), resetting.', numel(inliers), numObs);
        inliers = false(1, numObs);
    end
    
    inlierIndices = find(inliers);
    if isempty(inlierIndices)
        break;
    end
    if any(inlierIndices > numObs) || any(inlierIndices < 1)
        validMask = inlierIndices >= 1 & inlierIndices <= numObs;
        inliers = false(1, numObs);
        inliers(inlierIndices(validMask)) = true;
        inlierIndices = find(inliers);
        if isempty(inlierIndices) || nnz(inliers) < opts.MinInliers
            break;
        end
    end
    
    if numel(inliers) ~= size(obsWorld, 2)
        warning('trackAgainstStaticMap:DimensionMismatch', ...
            'Cannot proceed: inliers size (%d) != obsWorld columns (%d)', numel(inliers), size(obsWorld, 2));
        break;
    end

    if ~islogical(inliers) || numel(inliers) ~= size(obsWorld, 2)
        warning('trackAgainstStaticMap:InvalidInliersBeforeUse', ...
            'inliers invalid before use: logical=%d, size=%d, expected=%d', ...
            islogical(inliers), numel(inliers), size(obsWorld, 2));
        break;
    end
    
    inlierIdx = find(inliers);
    if isempty(inlierIdx) || numel(inlierIdx) == 0
        break;
    end
    
    if any(inlierIdx < 1) || any(inlierIdx > numel(idx))
        warning('trackAgainstStaticMap:InvalidInlierIdx', ...
            'inlierIdx out of range: [%d, %d], idx size: %d', ...
            min(inlierIdx), max(inlierIdx), numel(idx));
        break;
    end
    
    obsWorldInliers = obsWorld(:, inliers);
    corresNormalsInliers = corresNormals(:, inliers);
    mapPointsInliers = mapPositions(:, idx(inlierIdx));

    residuals = sum(corresNormalsInliers .* (obsWorldInliers - mapPointsInliers), 1);
    
    if ~exist('finalResiduals', 'var') || isempty(finalResiduals)
        finalResiduals = residuals;
    else
        finalResiduals = residuals;
    end

    numInliers = numel(residuals);
    if numInliers < 6
        break;
    end
    A = zeros(numInliers, 6);
    b = -residuals';

    for k = 1:numInliers
        pw = obsWorldInliers(:, k);
        n = corresNormalsInliers(:, k);
        if any(isnan(pw)) || any(isnan(n)) || any(isinf(pw)) || any(isinf(n))
            continue;
        end
        A(k, 1:3) = (cross(n, pw))';
        A(k, 4:6) = n';
    end

    validRows = ~any(isnan(A), 2) & ~any(isinf(A), 2) & ~isnan(b) & ~isinf(b);
    if nnz(validRows) < 6
        break;
    end
    A = A(validRows, :);
    b = b(validRows);

    try
        if opts.Damping > 0
            H = A' * A + opts.Damping * eye(6);
            g = A' * b;
            xi = H \ g;
        else
            xi = A \ b;
        end
        
        if any(isnan(xi)) || any(isinf(xi))
            break;
        end
    catch
        break;
    end

    if norm(xi) < 1e-6
        success = true;
        break;
    end

    Twc = se3Exp(xi) * Twc;

    if ~isempty(residuals) && numel(residuals) > 0
        rmse = sqrt(mean(residuals.^2));
    else
        rmse = NaN;
    end

    if opts.Verbose
        fprintf('[trackAgainstStaticMap] Iter %d: RMSE = %.4f, update norm = %.4e\n', ...
            iter, rmse, norm(xi));
    end
end

if isnan(rmse)
    if exist('finalResiduals', 'var') && ~isempty(finalResiduals) && numel(finalResiduals) > 0
        rmse = sqrt(mean(finalResiduals.^2));
    elseif exist('residuals', 'var') && ~isempty(residuals) && numel(residuals) > 0
        rmse = sqrt(mean(residuals.^2));
    else
        rmse = NaN;
    end
end

if ~exist('residuals', 'var') || isempty(residuals)
    if exist('finalResiduals', 'var') && ~isempty(finalResiduals)
        residuals = finalResiduals;
    else
        residuals = [];
    end
end

if nnz(inliers) >= opts.MinInliers
    success = true;
end

fullInlierMask = false(1, size(pointsCam, 2));
fullIdx = find(mask);
if ~isempty(fullIdx) && ~isempty(inliers) && numel(inliers) == numel(fullIdx)
    fullInlierMask(fullIdx(inliers)) = true;
end

fullCorrespondences = zeros(1, size(pointsCam, 2));
if ~isempty(fullIdx) && ~isempty(idx) && numel(idx) == numel(fullIdx)
    fullCorrespondences(fullIdx) = idx;
end

result = buildResult(Twc, success, iteration, rmse, fullInlierMask, fullCorrespondences, residuals);
end

function result = buildResult(Twc, success, iterations, rmse, inlierMask, correspondences, residuals)
result = struct('Twc', Twc, ...
    'success', logical(success), ...
    'iterations', iterations, ...
    'rmse', rmse, ...
    'inlierMask', inlierMask, ...
    'correspondences', correspondences, ...
    'residuals', residuals);
end

function T = se3Exp(xi)
omega = xi(1:3);
upsilon = xi(4:6);
theta = norm(omega);

Omega = skew(omega);
Omega2 = Omega * Omega;

if theta < 1e-8
    R = eye(3) + Omega;
    V = eye(3) + 0.5 * Omega;
else
    theta2 = theta * theta;
    theta3 = theta2 * theta;
    R = eye(3) + (sin(theta) / theta) * Omega + ((1 - cos(theta)) / theta2) * Omega2;
    V = eye(3) + ((1 - cos(theta)) / theta2) * Omega + ((theta - sin(theta)) / theta3) * Omega2;
end

t = V * upsilon;
T = eye(4);
T(1:3, 1:3) = R;
T(1:3, 4) = t;
end

function S = skew(v)
S = [   0   -v(3)  v(2);
      v(3)    0   -v(1);
     -v(2)  v(1)    0  ];
end

function normals = normalizeColumns(vectors)
norms = sqrt(sum(vectors.^2, 1));
norms(norms < eps) = 1;
normals = vectors ./ norms;
end

function validateStaticMap(map)
requiredFields = {'positions', 'normals', 'colours'};
for f = requiredFields
    if ~isfield(map, f{1})
        error('trackAgainstStaticMap:MissingField', ...
            'Map struct missing field ''%s''.', f{1});
    end
end
end

