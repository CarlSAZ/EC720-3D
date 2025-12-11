%% VISUALIZE_SEGMENTATION_ENERGY
% Visualize static/dynamic segmentation energy for a single frame

clear; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
startupPath = fullfile(scriptDir, 'startup.m');
if exist(startupPath, 'file') == 2
    run(startupPath);
else
    error('startup.m not found. Please run from the project root.');
end

%% Configuration
sequenceName = 'hotel_umd/maryland_hotel3';
refFrameIdx = 1;
segFrameIdx = 2;

depthFilterOpts = struct('minDepth', 0.4, 'maxDepth', 4.5, 'medianKernel', 3);
denoiseOpts = struct('radius', 0.04, 'minNeighbors', 12);
normalOpts = struct('K', 25);
mapInitOpts = struct( ...
    'DefaultConfidence', 3.0, ...
    'DefaultRadius', 0.04, ...
    'MergeRadius', 0.07, ...
    'ObservationWeight', 1.5, ...
    'MaxConfidence', 12.0, ...
    'ConfidenceDecay', 0.02);
segmentationOpts = struct( ...
    'GeometryThreshold', 0.05, ...
    'ColorThreshold', 0.18, ...
    'NormalThreshold', pi/5, ...
    'EnergyThreshold', 2.5, ...
    'EnergyWeights', struct('geometry', 1.2, 'colour', 1.0, 'normal', 0.7, 'prior', 1.0), ...
    'MinConfidence', 0.2);
sampleStep = 5;

%% Load reference frame and build static map
frameRef = loadFrame(sequenceName, refFrameIdx, depthFilterOpts, denoiseOpts, normalOpts);
XYZworld = frameRef.Rwc * frameRef.XYZcam + frameRef.twc;
normalsWorld = normalizeColumns(frameRef.Rwc * frameRef.normalsCam);
pcInit = struct('XYZ', XYZworld, 'RGB', frameRef.RGB);
mapInitArgs = structToNameValue(mapInitOpts);
staticMap = initializeStaticMap(pcInit, normalsWorld, mapInitArgs{:});

%% Load segmentation frame
frameSeg = loadFrame(sequenceName, segFrameIdx, depthFilterOpts, denoiseOpts, normalOpts);

sampleIdx = 1:sampleStep:size(frameSeg.XYZcam, 2);
pointsCam = frameSeg.XYZcam(:, sampleIdx);
coloursCam = frameSeg.RGB(:, sampleIdx);
normalsCam = frameSeg.normalsCam(:, sampleIdx);

%% Predict static map into segmentation frame
prediction = predictStaticMap(staticMap, frameSeg.Twc);

%% Run static/dynamic segmentation
segArgs = [structToNameValue(segmentationOpts), {'NormalsCam', normalsCam}];
segResult = estimateStaticMask(pointsCam, coloursCam, prediction, segArgs{:});

numStatic = nnz(segResult.staticMask);
numDynamic = nnz(segResult.dynamicMask);

fprintf('\n[VisualizeSegmentationEnergy] Frame %d segmentation summary\n', segFrameIdx);
fprintf('  Sampled points  : %d\n', numel(segResult.staticMask));
fprintf('  Static points   : %d\n', numStatic);
fprintf('  Dynamic points  : %d\n', numDynamic);
fprintf('  Static ratio    : %.1f%%%%\n', 100 * numStatic / max(1, numStatic + numDynamic));
    fprintf('  Energy (min/max): [%.3f, %.3f]\n', min(segResult.energy), max(segResult.energy));

%% Visualization
figure('Name', sprintf('Segmentation Energy (Frame %d)', segFrameIdx), 'Color', 'w', ...
    'Position', [100, 100, 1200, 600]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
hold on;
staticPts = pointsCam(:, segResult.staticMask);
dynamicPts = pointsCam(:, segResult.dynamicMask);
if ~isempty(staticPts)
    scatter3(staticPts(1, :), staticPts(2, :), staticPts(3, :), 12, [0, 0.8, 0], 'filled');
end
if ~isempty(dynamicPts)
    scatter3(dynamicPts(1, :), dynamicPts(2, :), dynamicPts(3, :), 12, [0.9, 0, 0], 'filled');
end
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
legend({'Static', 'Dynamic'}, 'Location', 'best');
axis equal; grid on; view(135, 30);
title(sprintf('Camera frame classification (Frame %d)', segFrameIdx));

nexttile;
energyStatic = segResult.energy(segResult.staticMask);
energyDynamic = segResult.energy(segResult.dynamicMask);
hold on;
if ~isempty(energyDynamic)
    histogram(energyDynamic, 60, 'FaceColor', [0.95, 0.4, 0.4], 'EdgeColor', 'none', 'DisplayName', 'Dynamic');
end
if ~isempty(energyStatic)
    histogram(energyStatic, 60, 'FaceColor', [0.4, 0.8, 0.4], 'EdgeColor', 'none', 'DisplayName', 'Static');
end
xlabel('Energy'); ylabel('Count');
title('Segmentation energy distribution');
legend('Location', 'best');
set(gca, 'YScale', 'log');

%% Helper functions
function frame = loadFrame(sequenceName, frameIdx, depthFilterOpts, denoiseOpts, normalOpts)
data = loadSUN3D(sequenceName, frameIdx);
if isempty(data.image) || isempty(data.depth)
    error('visualize_segmentation_energy:EmptyData', 'Failed to load frame %d.', frameIdx);
end

colorImg = imread(data.image{1});
depthImg = double(depthRead(data.depth{1}));

[XYZcam, RGB] = depthToPointCloud(colorImg, depthImg, data.K, 'depthFilter', depthFilterOpts);
if ~isempty(denoiseOpts)
    denoiseArgs = structToNameValue(denoiseOpts);
    [XYZcam, RGB] = denoisePointCloud(XYZcam, RGB, denoiseArgs{:});
end

normalArgs = structToNameValue(normalOpts);
normalsCam = estimateNormals(XYZcam, normalArgs{:}, 'ViewPoint', [0; 0; 0]);
if size(normalsCam, 2) ~= size(XYZcam, 2)
    warning('visualize_segmentation_energy:NormalMismatch', ...
        'Normal count mismatch, using default normals.');
    normalsCam = zeros(3, size(XYZcam, 2));
    normalsCam(3, :) = 1;
end

Twc = ensureHomogeneousTransform(data.extrinsicsC2W(:, :, frameIdx));
frame = struct('XYZcam', XYZcam, 'RGB', RGB, 'normalsCam', normalsCam, ...
    'Twc', Twc, 'Rwc', Twc(1:3, 1:3), 'twc', Twc(1:3, 4));
end

function args = structToNameValue(s)
if isempty(s)
    args = {};
    return;
end
fields = fieldnames(s);
args = cell(1, numel(fields) * 2);
for i = 1:numel(fields)
    args{2*i - 1} = fields{i};
    args{2*i} = s.(fields{i});
end
end

function normals = normalizeColumns(vectors)
norms = sqrt(sum(vectors.^2, 1));
norms(norms < eps) = 1;
normals = vectors ./ norms;
end

function T = ensureHomogeneousTransform(Traw)
if isempty(Traw)
    error('Empty transform encountered.');
end
sz = size(Traw);
if all(sz == [4, 4])
    T = Traw;
    return;
end
if all(sz == [3, 4])
    T = eye(4);
    T(1:3, 1:3) = Traw(1:3, 1:3);
    T(1:3, 4) = Traw(1:3, 4);
    return;
end
if all(sz == [3, 3])
    T = eye(4);
    T(1:3, 1:3) = Traw;
    return;
end
error('Unsupported transform size %dx%d.', sz(1), sz(2));
end
