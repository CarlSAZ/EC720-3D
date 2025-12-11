% Author: Anqi Wei
% Adapted from StaticFusion
% Reference: StaticFusion: Background Reconstruction for Dense RGB-D SLAM in Dynamic Environments
%            GitHub: https://github.com/raluca-scona/staticfusion

function result = trackFrameToFrame(prevPointsWorld, prevNormalsWorld, currentPointsCam, currentNormalsCam, initialPose, varargin)
% Track camera pose using frame-to-frame ICP

parser = inputParser;
addParameter(parser, 'MaxIterations', 20, @(x) validateattributes(x, {'double'}, {'scalar','integer','positive'}));
addParameter(parser, 'MaxCorrespondence', 0.15, @(x) validateattributes(x, {'double'}, {'scalar','positive'}));
addParameter(parser, 'NormalSimilarity', cosd(60), @(x) validateattributes(x, {'double'}, {'scalar','>=',-1,'<=',1}));
addParameter(parser, 'Damping', 1e-3, @(x) validateattributes(x, {'double'}, {'scalar','nonnegative'}));
addParameter(parser, 'MinInliers', 30, @(x) validateattributes(x, {'double'}, {'scalar','integer','>=',0}));
parse(parser, varargin{:});
opts = parser.Results;

Twc = initialPose;
success = false;
rmse = NaN;

if isempty(prevPointsWorld) || isempty(currentPointsCam)
    inlierMask = false(1, size(currentPointsCam, 2));
    result = buildResult(Twc, success, 0, rmse, inlierMask, [], []);
    return;
end

maxPoints = 5000;
prevIndices = 1:size(prevPointsWorld, 2);
currentIndices = 1:size(currentPointsCam, 2);

if size(prevPointsWorld, 2) > maxPoints
    idx = randperm(size(prevPointsWorld, 2), maxPoints);
    prevPointsWorld = prevPointsWorld(:, idx);
    prevNormalsWorld = prevNormalsWorld(:, idx);
    prevIndices = prevIndices(idx);
end
if size(currentPointsCam, 2) > maxPoints
    idx = randperm(size(currentPointsCam, 2), maxPoints);
    currentPointsCam = currentPointsCam(:, idx);
    currentNormalsCam = currentNormalsCam(:, idx);
    currentIndices = currentIndices(idx);
end

inlierMask = false(1, size(currentPointsCam, 2));

for iter = 1:opts.MaxIterations
    R = Twc(1:3, 1:3);
    t = Twc(1:3, 4);
    currentPointsWorld = R * currentPointsCam + t;
    currentNormalsWorld = R * currentNormalsCam;
    
    try
        [idx, dists] = knnsearch(prevPointsWorld', currentPointsWorld');
    catch
        break;
    end
    
    validDistance = (dists <= opts.MaxCorrespondence);
    
    corresNormals = prevNormalsWorld(:, idx);
    normalSimilarity = sum(corresNormals .* currentNormalsWorld, 1);
    if size(normalSimilarity, 1) > 1
        normalSimilarity = normalSimilarity';
    end
    validNormals = (normalSimilarity >= opts.NormalSimilarity);
    
    inliers = validDistance & validNormals;
    
    if size(inliers, 1) > size(inliers, 2)
        inliers = inliers';
    end
    if numel(inliers) ~= size(currentPointsWorld, 2)
        warning('trackFrameToFrame: Inlier size mismatch, resetting.');
        inliers = false(1, size(currentPointsWorld, 2));
    end
    
    if nnz(inliers) < opts.MinInliers
        break;
    end
    
    validIdx = idx >= 1 & idx <= size(prevPointsWorld, 2);
    if ~all(validIdx(inliers))
        warning('trackFrameToFrame: Invalid correspondence indices, skipping iteration.');
        break;
    end
    
    P = currentPointsWorld(:, inliers);
    Q = prevPointsWorld(:, idx(inliers));
    
    P_mean = mean(P, 2);
    Q_mean = mean(Q, 2);
    P_centered = P - P_mean;
    Q_centered = Q - Q_mean;
    
    H = P_centered * Q_centered';
    [U, ~, V] = svd(H);
    R_update = V * U';
    if det(R_update) < 0
        V(:, end) = -V(:, end);
        R_update = V * U';
    end
    
    t_update = Q_mean - R_update * P_mean;
    
    Twc_new = [R_update, t_update; 0, 0, 0, 1] * Twc;
    
    relChange = norm(Twc_new(1:3, 4) - Twc(1:3, 4));
    if relChange < 1e-4
        Twc = Twc_new;
        success = true;
        break;
    end
    
    Twc = Twc_new;
end

R = Twc(1:3, 1:3);
t = Twc(1:3, 4);
currentPointsWorld = R * currentPointsCam + t;

try
    [idx, dists] = knnsearch(prevPointsWorld', currentPointsWorld');
catch ME
    warning('trackFrameToFrame: knnsearch failed in final evaluation: %s', ME.message);
    result = buildResult(Twc, success, 0, rmse, inlierMask, [], []);
    return;
end

if any(idx < 1) || any(idx > size(prevPointsWorld, 2))
    warning('trackFrameToFrame: Invalid indices in final evaluation.');
    idx(idx < 1) = 1;
    idx(idx > size(prevPointsWorld, 2)) = size(prevPointsWorld, 2);
end

validDistance = (dists <= opts.MaxCorrespondence);
corresNormals = prevNormalsWorld(:, idx);
currentNormalsWorld = R * currentNormalsCam;
normalSimilarity = sum(corresNormals .* currentNormalsWorld, 1);
if size(normalSimilarity, 1) > 1
    normalSimilarity = normalSimilarity';
end
validNormals = (normalSimilarity >= opts.NormalSimilarity);

if size(validDistance, 1) > size(validDistance, 2)
    validDistance = validDistance';
end
if size(validNormals, 1) > size(validNormals, 2)
    validNormals = validNormals';
end
if numel(validDistance) ~= numel(validNormals)
    minLen = min(numel(validDistance), numel(validNormals));
    validDistance = validDistance(1:minLen);
    validNormals = validNormals(1:minLen);
end

inlierMask = validDistance & validNormals;

if nnz(inlierMask) >= opts.MinInliers
    success = true;
    rmse = sqrt(mean(dists(inlierMask).^2));
else
    success = false;
    rmse = NaN;
end

result = buildResult(Twc, success, nnz(inlierMask), rmse, inlierMask, idx, dists);

end

function result = buildResult(Twc, success, numInliers, rmse, inlierMask, correspondences, residuals)
result = struct();
result.Twc = Twc;
result.success = success;
result.numInliers = numInliers;
result.rmse = rmse;
result.inlierMask = inlierMask;
result.correspondences = correspondences;
result.residuals = residuals;
end
