%% VISUALIZE_INITIALIZE_STATIC_MAP
% Visualize the complete process of initializing a static map from point cloud data.
% This script demonstrates each step of initializeStaticMap.m with detailed visualizations.
%
% Usage: Run this script directly in MATLAB. It will:
%   1. Load a sample frame from SUN3D
%   2. Show the input point cloud and normals
%   3. Visualize each step of the initialization process
%   4. Display the final surfel map with all attributes
%
% Author: Generated for EC720-3D project
% Date: 2024

clear; close all; clc;

fprintf('=== Visualizing initializeStaticMap Process ===\n\n');

%% Setup paths
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
startupPath = fullfile(projectRoot, 'startup.m');
if exist(startupPath, 'file') == 2
    run(startupPath);
else
    error('startup.m not found. Please run from the project root.');
end

%% Configuration
sequenceName = 'hotel_umd/maryland_hotel3';
frameIdx = 1;

% Preprocessing parameters (matching staticfusion_demo.m)
depthFilterOpts = struct('minDepth', 0.4, 'maxDepth', 4.5, 'medianKernel', 3);
denoiseOpts = struct('radius', 0.04, 'minNeighbors', 12);
normalOpts = struct('K', 25);
processing.downsample = 5;  % Downsample for visualization

% Map initialization parameters
mapInitOpts = struct( ...
    'DefaultConfidence', 3.0, ...
    'DefaultRadius', 0.04, ...
    'MergeRadius', 0.07, ...
    'ObservationWeight', 1.5, ...
    'MaxConfidence', 12.0, ...
    'ConfidenceDecay', 0.02);

fprintf('Configuration:\n');
fprintf('  Sequence: %s, Frame: %d\n', sequenceName, frameIdx);
fprintf('  DefaultConfidence: %.1f\n', mapInitOpts.DefaultConfidence);
fprintf('  DefaultRadius: %.3f m\n', mapInitOpts.DefaultRadius);
fprintf('  Downsample: %d\n\n', processing.downsample);

%% Step 1: Load and preprocess data
fprintf('Step 1: Loading and preprocessing data...\n');
data = loadSUN3D(sequenceName, frameIdx);

if isempty(data.image) || isempty(data.depth)
    error('Failed to load image/depth for frame %d.', frameIdx);
end

image = imread(data.image{1});
depth = double(depthRead(data.depth{1}));

[XYZcam, RGB] = depthToPointCloud(image, depth, data.K, 'depthFilter', depthFilterOpts);
[XYZcam, RGB] = denoisePointCloud(XYZcam, RGB, 'radius', denoiseOpts.radius, ...
    'minNeighbors', denoiseOpts.minNeighbors);

normalsCam = estimateNormals(XYZcam, 'K', normalOpts.K, 'ViewPoint', [0; 0; 0]);

% Transform to world coordinates
firstPose = ensureHomogeneousTransform(data.extrinsicsC2W(:, :, frameIdx));
Rw = firstPose(1:3, 1:3);
tw = firstPose(1:3, 4);

XYZworld = Rw * XYZcam + tw;
normalsWorld = normalizeColumns(Rw * normalsCam);

% Downsample for visualization
sampleIdx = 1:processing.downsample:size(XYZworld, 2);
XYZworldSample = XYZworld(:, sampleIdx);
RGBsample = RGB(:, sampleIdx);
normalsWorldSample = normalsWorld(:, sampleIdx);

fprintf('  Loaded %d points (showing %d for visualization)\n\n', ...
    size(XYZworld, 2), size(XYZworldSample, 2));

%% Step 2: Prepare input for initializeStaticMap
fprintf('Step 2: Preparing input for initializeStaticMap...\n');
pcInit = struct('XYZ', XYZworld, 'RGB', RGB);
mapInitArgs = structToNameValue(mapInitOpts);

fprintf('  Input point cloud: %d points\n', size(XYZworld, 2));
fprintf('  Input normals: %d normals\n', size(normalsWorld, 2));
fprintf('  Input colors: %d colors\n\n', size(RGB, 2));

%% Step 3: Call initializeStaticMap
fprintf('Step 3: Calling initializeStaticMap...\n');
staticMap = initializeStaticMap(pcInit, normalsWorld, mapInitArgs{:});

fprintf('  Created static map with %d surfels\n', size(staticMap.positions, 2));
fprintf('  Confidence range: [%.2f, %.2f]\n', ...
    min(staticMap.confidence), max(staticMap.confidence));
fprintf('  Radius: %.3f m (all surfels)\n', staticMap.radius(1));
fprintf('  Parameters saved: mergeRadius=%.3f, observationWeight=%.1f\n\n', ...
    staticMap.params.mergeRadius, staticMap.params.observationWeight);

%% Step 4: Create separate visualizations for each step
fprintf('Step 4: Creating separate visualizations for each step...\n');

mapSampleIdx = 1:processing.downsample:size(staticMap.positions, 2);

%% Figure 1: Original RGB Image
fig1 = figure('Name', 'Step 1: Original RGB Image', ...
    'Position', [50, 50, 800, 600], 'Color', 'w');
imshow(image);
title('Step 1: Original RGB Image', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(sprintf('Frame %d from %s', frameIdx, sequenceName), 'FontSize', 12);
fprintf('  Created Figure 1: Original RGB Image\n');

%% Figure 2: Input Point Cloud (World Coordinates)
fig2 = figure('Name', 'Step 2: Input Point Cloud', ...
    'Position', [100, 100, 900, 700], 'Color', 'w');
hold on;
scatter3(XYZworldSample(1, :), XYZworldSample(2, :), XYZworldSample(3, :), ...
    10, RGBsample', '.');
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title('Step 2: Input Point Cloud (World Coordinates)', ...
    'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
c = colorbar;
c.Label.String = 'RGB Color';
fprintf('  Created Figure 2: Input Point Cloud\n');

%% Figure 3: Input Normals Visualization
fig3 = figure('Name', 'Step 3: Input Normals', ...
    'Position', [150, 150, 900, 700], 'Color', 'w');
hold on;
% Show points
scatter3(XYZworldSample(1, :), XYZworldSample(2, :), XYZworldSample(3, :), ...
    5, 'b', '.');
% Show normals (every 10th for clarity)
normalSample = 1:10:size(normalsWorldSample, 2);
quiver3(XYZworldSample(1, normalSample), ...
        XYZworldSample(2, normalSample), ...
        XYZworldSample(3, normalSample), ...
        normalsWorldSample(1, normalSample) * 0.1, ...
        normalsWorldSample(2, normalSample) * 0.1, ...
        normalsWorldSample(3, normalSample) * 0.1, ...
        'r', 'LineWidth', 1.5, 'AutoScale', 'off');
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title('Step 3: Input Normals (Red Arrows)', ...
    'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
legend('Points', 'Normals', 'Location', 'best', 'FontSize', 11);
fprintf('  Created Figure 3: Input Normals\n');

%% Figure 4: Surfel Positions (map.positions)
fig4 = figure('Name', 'Step 4: Surfel Positions', ...
    'Position', [200, 200, 900, 700], 'Color', 'w');
scatter3(staticMap.positions(1, mapSampleIdx), ...
         staticMap.positions(2, mapSampleIdx), ...
         staticMap.positions(3, mapSampleIdx), ...
         10, staticMap.colours(:, mapSampleIdx)', '.');
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title('Step 4: Surfel Positions (map.positions)', ...
    'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
c = colorbar;
c.Label.String = 'RGB Color';
fprintf('  Created Figure 4: Surfel Positions\n');

%% Figure 5: Surfel Normals (normalized)
fig5 = figure('Name', 'Step 5: Surfel Normals', ...
    'Position', [250, 250, 900, 700], 'Color', 'w');
hold on;
scatter3(staticMap.positions(1, mapSampleIdx), ...
         staticMap.positions(2, mapSampleIdx), ...
         staticMap.positions(3, mapSampleIdx), ...
         5, 'b', '.');
normalMapSample = 1:10:length(mapSampleIdx);
normalMapIdx = mapSampleIdx(normalMapSample);
quiver3(staticMap.positions(1, normalMapIdx), ...
        staticMap.positions(2, normalMapIdx), ...
        staticMap.positions(3, normalMapIdx), ...
        staticMap.normals(1, normalMapIdx) * 0.1, ...
        staticMap.normals(2, normalMapIdx) * 0.1, ...
        staticMap.normals(3, normalMapIdx) * 0.1, ...
        'g', 'LineWidth', 1.5, 'AutoScale', 'off');
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title('Step 5: Surfel Normals (Normalized, Green Arrows)', ...
    'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
legend('Surfels', 'Normals', 'Location', 'best', 'FontSize', 11);
fprintf('  Created Figure 5: Surfel Normals\n');

%% Figure 6: Surfel Colors
fig6 = figure('Name', 'Step 6: Surfel Colors', ...
    'Position', [300, 300, 900, 700], 'Color', 'w');
scatter3(staticMap.positions(1, mapSampleIdx), ...
         staticMap.positions(2, mapSampleIdx), ...
         staticMap.positions(3, mapSampleIdx), ...
         10, staticMap.colours(:, mapSampleIdx)', '.');
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title('Step 6: Surfel Colors (map.colours)', ...
    'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
c = colorbar;
c.Label.String = 'RGB Color';
fprintf('  Created Figure 6: Surfel Colors\n');

%% Figure 7: Surfel Confidence
fig7 = figure('Name', 'Step 7: Surfel Confidence', ...
    'Position', [350, 350, 900, 700], 'Color', 'w');
confSample = staticMap.confidence(mapSampleIdx);
scatter3(staticMap.positions(1, mapSampleIdx), ...
         staticMap.positions(2, mapSampleIdx), ...
         staticMap.positions(3, mapSampleIdx), ...
         20, confSample', 'filled');
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title(sprintf('Step 7: Surfel Confidence (map.confidence, all=%.1f)', ...
    staticMap.confidence(1)), 'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
c = colorbar;
c.Label.String = 'Confidence';
colormap(gca, 'hot');
fprintf('  Created Figure 7: Surfel Confidence\n');

%% Figure 8: Surfel Radius
fig8 = figure('Name', 'Step 8: Surfel Radius', ...
    'Position', [400, 400, 900, 700], 'Color', 'w');
radiusSample = staticMap.radius(mapSampleIdx);
scatter3(staticMap.positions(1, mapSampleIdx), ...
         staticMap.positions(2, mapSampleIdx), ...
         staticMap.positions(3, mapSampleIdx), ...
         20, radiusSample' * 100, 'filled');  % Scale for visualization
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title(sprintf('Step 8: Surfel Radius (map.radius, all=%.3f m)', ...
    staticMap.radius(1)), 'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
c = colorbar;
c.Label.String = 'Radius (cm)';
colormap(gca, 'cool');
fprintf('  Created Figure 8: Surfel Radius\n');

%% Figure 9: Complete Surfel Map with All Attributes
fig9 = figure('Name', 'Step 9: Complete Surfel Map', ...
    'Position', [450, 450, 900, 700], 'Color', 'w');
hold on;
% Color by RGB
scatter3(staticMap.positions(1, mapSampleIdx), ...
         staticMap.positions(2, mapSampleIdx), ...
         staticMap.positions(3, mapSampleIdx), ...
         15, staticMap.colours(:, mapSampleIdx)', '.');
% Show some normals
normalFinalSample = 1:20:length(mapSampleIdx);
normalFinalIdx = mapSampleIdx(normalFinalSample);
quiver3(staticMap.positions(1, normalFinalIdx), ...
        staticMap.positions(2, normalFinalIdx), ...
        staticMap.positions(3, normalFinalIdx), ...
        staticMap.normals(1, normalFinalIdx) * 0.08, ...
        staticMap.normals(2, normalFinalIdx) * 0.08, ...
        staticMap.normals(3, normalFinalIdx) * 0.08, ...
        'k', 'LineWidth', 1, 'AutoScale', 'off');
axis equal; grid on;
xlabel('X (m)', 'FontSize', 12); 
ylabel('Y (m)', 'FontSize', 12); 
zlabel('Z (m)', 'FontSize', 12);
title('Step 9: Complete Surfel Map (Positions + Colors + Normals)', ...
    'FontSize', 14, 'FontWeight', 'bold');
view(135, 30);
legend('Surfels', 'Normals', 'Location', 'best', 'FontSize', 11);
c = colorbar;
c.Label.String = 'RGB Color';
fprintf('  Created Figure 9: Complete Surfel Map\n');

%% Print summary information
fprintf('\n=== Summary ===\n');
fprintf('Input Data:\n');
fprintf('  Point cloud size: %d points\n', size(XYZworld, 2));
fprintf('  Normals size: %d normals\n', size(normalsWorld, 2));
fprintf('  Colors size: %d colors\n', size(RGB, 2));

fprintf('\nOutput Map Structure:\n');
fprintf('  map.positions: %dx%d (3xN)\n', size(staticMap.positions, 1), ...
    size(staticMap.positions, 2));
fprintf('  map.normals: %dx%d (3xN, normalized)\n', size(staticMap.normals, 1), ...
    size(staticMap.normals, 2));
fprintf('  map.colours: %dx%d (3xN)\n', size(staticMap.colours, 1), ...
    size(staticMap.colours, 2));
fprintf('  map.confidence: 1x%d (all = %.1f)\n', size(staticMap.confidence, 2), ...
    staticMap.confidence(1));
fprintf('  map.radius: 1x%d (all = %.3f m)\n', size(staticMap.radius, 2), ...
    staticMap.radius(1));
fprintf('  map.params: struct with %d fields\n', numel(fieldnames(staticMap.params)));

fprintf('\nKey Points:\n');
fprintf('  ✓ Each surfel has: position, normal, color, confidence, radius\n');
fprintf('  ✓ All fields are column-aligned (column i = surfel i)\n');
fprintf('  ✓ Normals are normalized to unit length\n');
fprintf('  ✓ Confidence and radius are initialized uniformly\n');
fprintf('  ✓ Parameters are saved for future updates\n');

fprintf('\n=== Visualization Complete ===\n');
fprintf('Generated 9 separate figures:\n');
fprintf('  Figure 1: Original RGB Image\n');
fprintf('  Figure 2: Input Point Cloud\n');
fprintf('  Figure 3: Input Normals\n');
fprintf('  Figure 4: Surfel Positions\n');
fprintf('  Figure 5: Surfel Normals\n');
fprintf('  Figure 6: Surfel Colors\n');
fprintf('  Figure 7: Surfel Confidence\n');
fprintf('  Figure 8: Surfel Radius\n');
fprintf('  Figure 9: Complete Surfel Map\n');
fprintf('\nEach figure shows a different aspect of the initialization process.\n');
fprintf('You can examine each figure individually for detailed analysis.\n');

%% Helper functions
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

function normals = normalizeColumns(vectors)
    norms = sqrt(sum(vectors.^2, 1));
    norms(norms < eps) = 1;
    normals = vectors ./ norms;
end
