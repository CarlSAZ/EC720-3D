%% CHANGE_DETECTION_DEMO
% Simple 3D change-detection pipeline using SUN3D frames

clear; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
utilsDir = fullfile(scriptDir, 'utils');
if exist(utilsDir, 'dir')
    addpath(utilsDir);
end

%% User configuration
sequenceName = 'hotel_umd/maryland_hotel3';
frameIDs = [1, 10];
skipColorDownsample = 5;

depthFilterOpts = struct('minDepth', 0.4, 'maxDepth', 4.5, 'medianKernel', 3);
denoiseOpts = struct('radius', 0.04, 'minNeighbors', 8);
diffParams = struct('threshold', 0.05, 'requireMutual', true);
clusterOpts = struct('radius', 0.12, 'minClusterSize', 40);
visualOpts = struct('downsample', skipColorDownsample, ...
    'showLegend', true, ...
    'markerSize', 14, ...
    'savePath', "");
visualViews = {[135, 30], [0, 90]};

assert(numel(frameIDs) == 2, 'Provide exactly two frame indices to compare.');

%% Load data
fprintf('Loading SUN3D data (%s) for frames %d and %d...\n', ...
    sequenceName, frameIDs(1), frameIDs(2));
data = loadSUN3D(sequenceName, frameIDs);

if isempty(data.extrinsicsC2W)
    error('Extrinsic camera poses are missing for the requested sequence.');
end

pcA = buildWorldPointCloud(data, frameIDs, 1, depthFilterOpts, denoiseOpts);
pcB = buildWorldPointCloud(data, frameIDs, 2, depthFilterOpts, denoiseOpts);

fprintf('Frame %d produced %d valid points; frame %d produced %d valid points.\n', ...
    pcA.frameID,     size(pcA.XYZ, 2), pcB.frameID, size(pcB.XYZ, 2));

%% Change detection
fprintf('Running nearest-neighbour differencing with threshold = %.2f m...\n', diffParams.threshold);

diffArgs = structToNameValue(diffParams);
diffResult = computePointCloudDiff(pcA, pcB, diffArgs{:});

addedXYZ = diffResult.addedXYZ;
removedXYZ = diffResult.removedXYZ;

if ~isempty(addedXYZ)
    clusterArgs = structToNameValue(clusterOpts);
    [keepAdded, ~] = filterClusters(addedXYZ', clusterArgs{:});
    addedXYZ = addedXYZ(:, keepAdded(:)');
end
if ~isempty(removedXYZ)
    clusterArgs = structToNameValue(clusterOpts);
    [keepRemoved, ~] = filterClusters(removedXYZ', clusterArgs{:});
    removedXYZ = removedXYZ(:, keepRemoved(:)');
end

fprintf('Raw detections: %d added, %d removed.\n', ...
    size(diffResult.addedXYZ, 2), size(diffResult.removedXYZ, 2));
fprintf('After cluster filtering: %d added, %d removed.\n', ...
    size(addedXYZ, 2), size(removedXYZ, 2));

%% Visualization
figure('Name', 'SUN3D Change Detection Demo');

subplot(1, 2, 1);
title(sprintf('Frame %d (World Coordinates)', pcA.frameID));
hold on; grid on; axis equal;
scatterPointCloud(pcA.XYZ, pcA.RGB, skipColorDownsample);
view(135, 30);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');

subplot(1, 2, 2);
title(sprintf('Detected Changes (Frames %d vs %d)', pcA.frameID, pcB.frameID));
hold on; grid on; axis equal;
scatterPointCloud(pcB.XYZ, pcB.RGB, skipColorDownsample);
if ~isempty(addedXYZ)
    scatter3(addedXYZ(1, :), addedXYZ(2, :), addedXYZ(3, :), visualOpts.markerSize, 'r', 'filled');
end
if ~isempty(removedXYZ)
    scatter3(removedXYZ(1, :), removedXYZ(2, :), removedXYZ(3, :), visualOpts.markerSize, 'b', 'filled');
end
legendEntries = {'Frame B points'};
if ~isempty(addedXYZ)
    legendEntries{end + 1} = 'Added (filtered)';
end
if ~isempty(removedXYZ)
    legendEntries{end + 1} = 'Removed (filtered)';
end
legend(legendEntries, 'Location', 'best');
view(135, 30);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');

visualArgs = structToNameValue(visualOpts);
visualizeChanges(pcA, pcB, addedXYZ, removedXYZ, visualArgs{:}, 'views', visualViews);

function pc = buildWorldPointCloud(data, frameIDsFull, localIdx, depthOpts, denoiseOpts)
image = imread(data.image{localIdx});
depth = double(depthRead(data.depth{localIdx}));

[XYZ, RGB] = depthToPointCloud(image, depth, data.K, 'depthFilter', depthOpts);

if ~isempty(denoiseOpts)
    denoiseArgs = structToNameValue(denoiseOpts);
    [XYZ, RGB] = denoisePointCloud(XYZ, RGB, denoiseArgs{:});
end

frameIDFull = frameIDsFull(localIdx);
Rt = data.extrinsicsC2W(:, :, frameIDFull);

XYZworld = transformPointCloud(XYZ, Rt);

pc = struct('XYZ', XYZworld, 'RGB', RGB, 'frameID', frameIDFull);
end

function scatterPointCloud(points, colours, downsample)
if nargin < 3 || downsample <= 1
    sampleIdx = 1:size(points, 2);
else
    sampleIdx = 1:downsample:size(points, 2);
end
scatter3(points(1, sampleIdx), points(2, sampleIdx), points(3, sampleIdx), ...
    3, colours(:, sampleIdx)', '.');
end

function args = structToNameValue(s)
if isempty(s)
    args = {};
    return;
end
if ~isstruct(s) || numel(s) ~= 1
    error('Expected a scalar struct for name/value conversion.');
end
fields = fieldnames(s);
args = cell(1, numel(fields) * 2);
for idx = 1:numel(fields)
    args{2*idx - 1} = fields{idx};
    args{2*idx} = s.(fields{idx});
end
end

