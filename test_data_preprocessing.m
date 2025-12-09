%% DATA LOADING AND PREPROCESSING TEST SCRIPT
% This script tests the complete data loading and preprocessing pipeline
% with visualization at each step to verify correctness.

clear; close all; clc;

fprintf('=== Data Loading and Preprocessing Test ===\n\n');

%% Step 0: Setup paths
scriptDir = fileparts(mfilename('fullpath'));
if exist('loadSUN3D', 'file') ~= 2
    startupPath = fullfile(scriptDir, 'startup.m');
    if exist(startupPath, 'file')
        run(startupPath);
    else
        error('startup.m is missing.');
    end
end

%% Step 1: Configuration (matching staticfusion_demo.m)
sequenceName = 'hotel_umd/maryland_hotel3';
depthFilterOpts = struct('minDepth', 0.4, 'maxDepth', 4.5, 'medianKernel', 3);
denoiseOpts = struct('radius', 0.04, 'minNeighbors', 12);
normalOpts = struct('K', 25);
processing.downsample = 5;

fprintf('Configuration:\n');
fprintf('  Sequence: %s\n', sequenceName);
fprintf('  Depth filter: min=%.1fm, max=%.1fm, median=%d\n', ...
    depthFilterOpts.minDepth, depthFilterOpts.maxDepth, depthFilterOpts.medianKernel);
fprintf('  Denoise: radius=%.2fm, minNeighbors=%d\n', ...
    denoiseOpts.radius, denoiseOpts.minNeighbors);
fprintf('  Normals: K=%d\n', normalOpts.K);
fprintf('  Downsample: %d\n\n', processing.downsample);

%% Step 2: Load SUN3D sequence data
fprintf('Step 1: Loading SUN3D sequence data...\n');
try
    data = loadSUN3D(sequenceName, 1);  % Load first frame only
    fprintf('  [OK] Successfully loaded frame 1\n');
    fprintf('    Image path: %s\n', data.image{1});
    fprintf('    Depth path: %s\n', data.depth{1});
    fprintf('    Camera intrinsics K:\n');
    fprintf('      [%.2f  %.2f  %.2f]\n', data.K(1,1), data.K(1,2), data.K(1,3));
    fprintf('      [%.2f  %.2f  %.2f]\n', data.K(2,1), data.K(2,2), data.K(2,3));
    fprintf('      [%.2f  %.2f  %.2f]\n', data.K(3,1), data.K(3,2), data.K(3,3));
catch ME
    fprintf('  [FAILED] %s\n', ME.message);
    return;
end

%% Step 3: Read RGB image and depth map
fprintf('\nStep 2: Reading RGB image and depth map...\n');
try
    image = imread(data.image{1});
    depth_raw = double(depthRead(data.depth{1}));
    fprintf('  [OK] Successfully read\n');
    fprintf('    Image size: %d x %d x %d\n', size(image, 1), size(image, 2), size(image, 3));
    fprintf('    Depth map size: %d x %d\n', size(depth_raw, 1), size(depth_raw, 2));
    valid_depth = depth_raw > 0;
    fprintf('    Valid depth pixels: %d / %d (%.1f%%)\n', ...
        nnz(valid_depth), numel(depth_raw), 100*nnz(valid_depth)/numel(depth_raw));
    if nnz(valid_depth) > 0
        fprintf('    Depth range: %.3f to %.3f meters\n', ...
            min(depth_raw(valid_depth)), max(depth_raw(valid_depth)));
    end
catch ME
    fprintf('  [FAILED] %s\n', ME.message);
    return;
end

% Visualization: Step 2
figure('Name', 'Step 2: Raw RGB Image and Depth Map', 'Position', [100, 100, 1000, 400]);
subplot(1, 2, 1);
imshow(image);
title('Raw RGB Image');
subplot(1, 2, 2);
imagesc(depth_raw);
axis image;  % Keep aspect ratio same as original image
axis off;    % Remove axes
colorbar;
title('Raw Depth Map (meters)');
colormap(gca, 'jet');
fprintf('  [VISUALIZATION] Figure 1: Raw RGB and depth map\n');

%% Step 4: Convert depth to point cloud (includes depth filtering)
fprintf('\nStep 3: Converting depth to point cloud (with depth filtering)...\n');
try
    [XYZcam, RGB] = depthToPointCloud(image, depth_raw, data.K, 'depthFilter', depthFilterOpts);
    fprintf('  [OK] Successfully converted\n');
    fprintf('    Point cloud size: %d points\n', size(XYZcam, 2));
    if size(XYZcam, 2) > 0
        fprintf('    X range: %.3f to %.3f meters\n', min(XYZcam(1,:)), max(XYZcam(1,:)));
        fprintf('    Y range: %.3f to %.3f meters\n', min(XYZcam(2,:)), max(XYZcam(2,:)));
        fprintf('    Z range: %.3f to %.3f meters\n', min(XYZcam(3,:)), max(XYZcam(3,:)));
    end
catch ME
    fprintf('  [FAILED] %s\n', ME.message);
    return;
end

% Visualization: Step 3
figure('Name', 'Step 3: Point Cloud (Camera Coordinates)', 'Position', [150, 150, 800, 600]);
if size(XYZcam, 2) > 0
    % Downsample for visualization
    viz_idx = 1:10:size(XYZcam, 2);
    scatter3(XYZcam(1, viz_idx), XYZcam(2, viz_idx), XYZcam(3, viz_idx), ...
        10, RGB(:, viz_idx)', 'filled');
    xlabel('X (meters)'); ylabel('Y (meters)'); zlabel('Z (meters)');
    title('Point Cloud in Camera Coordinates (view from camera, matching RGB image)');
    axis equal;
    grid on;
    colorbar;
    % Set camera view: view from behind camera looking along -Z direction
    % This matches the RGB image orientation (X right, Y down, Z forward)
    view(0, -90);  % Azimuth=0 (along X), Elevation=-90 (from above, looking down)
end
fprintf('  [VISUALIZATION] Figure 2: Point cloud in camera coordinates\n');

%% Step 5: Denoise point cloud
fprintf('\nStep 4: Denoising point cloud...\n');
try
    numPoints_before = size(XYZcam, 2);
    [XYZcam, RGB] = denoisePointCloud(XYZcam, RGB, 'radius', denoiseOpts.radius, ...
        'minNeighbors', denoiseOpts.minNeighbors);
    numPoints_after = size(XYZcam, 2);
    fprintf('  [OK] Successfully denoised\n');
    fprintf('    Points before: %d\n', numPoints_before);
    fprintf('    Points after: %d\n', numPoints_after);
    fprintf('    Removed: %d points (%.1f%%)\n', ...
        numPoints_before - numPoints_after, ...
        100*(numPoints_before - numPoints_after)/numPoints_before);
catch ME
    fprintf('  [FAILED] %s\n', ME.message);
    return;
end

% Visualization: Step 4
figure('Name', 'Step 4: Denoised Point Cloud', 'Position', [200, 200, 800, 600]);
if size(XYZcam, 2) > 0
    viz_idx = 1:10:size(XYZcam, 2);
    scatter3(XYZcam(1, viz_idx), XYZcam(2, viz_idx), XYZcam(3, viz_idx), ...
        10, RGB(:, viz_idx)', 'filled');
    xlabel('X (meters)'); ylabel('Y (meters)'); zlabel('Z (meters)');
    title('Denoised Point Cloud (view from camera, matching RGB image)');
    axis equal;
    grid on;
    colorbar;
    % Set camera view: view from behind camera looking along -Z direction
    view(0, -90);  % Azimuth=0 (along X), Elevation=-90 (from above, looking down)
end
fprintf('  [VISUALIZATION] Figure 3: Denoised point cloud\n');

%% Step 6: Estimate normals
fprintf('\nStep 5: Estimating surface normals...\n');
try
    normalsCam = estimateNormals(XYZcam, 'K', normalOpts.K, 'ViewPoint', [0; 0; 0]);
    fprintf('  [OK] Successfully estimated\n');
    fprintf('    Normal count: %d\n', size(normalsCam, 2));
    if size(normalsCam, 2) > 0
        fprintf('    Normal example (first 3 points):\n');
        for i = 1:min(3, size(normalsCam, 2))
            fprintf('      Point %d: [%.3f, %.3f, %.3f]\n', ...
                i, normalsCam(1,i), normalsCam(2,i), normalsCam(3,i));
        end
    end
catch ME
    fprintf('  [FAILED] %s\n', ME.message);
    return;
end

% Visualization: Step 5
figure('Name', 'Step 5: Point Cloud with Normals', 'Position', [250, 250, 800, 600]);
if size(XYZcam, 2) > 0
    viz_idx = 1:50:size(XYZcam, 2);  % More sparse for normals visualization
    scatter3(XYZcam(1, viz_idx), XYZcam(2, viz_idx), XYZcam(3, viz_idx), ...
        20, RGB(:, viz_idx)', 'filled');
    hold on;
    % Draw normals
    scale = 0.05;  % Scale factor for normal vectors
    quiver3(XYZcam(1, viz_idx), XYZcam(2, viz_idx), XYZcam(3, viz_idx), ...
        normalsCam(1, viz_idx)*scale, normalsCam(2, viz_idx)*scale, normalsCam(3, viz_idx)*scale, ...
        'r', 'LineWidth', 1);
    hold off;
    xlabel('X (meters)'); ylabel('Y (meters)'); zlabel('Z (meters)');
    title('Point Cloud with Surface Normals (view from camera, matching RGB image)');
    axis equal;
    grid on;
    legend('Points', 'Normals', 'Location', 'best');
    % Set camera view: view from behind camera looking along -Z direction
    view(0, -90);  % Azimuth=0 (along X), Elevation=-90 (from above, looking down)
end
fprintf('  [VISUALIZATION] Figure 4: Point cloud with normals\n');

%% Step 7: Data sampling (downsampling)
fprintf('\nStep 6: Data sampling (downsampling)...\n');
try
    numPoints_before = size(XYZcam, 2);
    sampleIdx = selectSampleIndices(size(XYZcam, 2), processing.downsample);
    XYZcamSample = XYZcam(:, sampleIdx);
    RGBsample = RGB(:, sampleIdx);
    normalsCamSample = normalsCam(:, sampleIdx);
    numPoints_after = size(XYZcamSample, 2);
    fprintf('  [OK] Successfully sampled\n');
    fprintf('    Points before: %d\n', numPoints_before);
    fprintf('    Points after: %d (downsample=%d)\n', numPoints_after, processing.downsample);
    fprintf('    Reduction: %.1f%%\n', 100*(1 - numPoints_after/numPoints_before));
catch ME
    fprintf('  [FAILED] %s\n', ME.message);
    return;
end

% Visualization: Step 6
figure('Name', 'Step 6: Downsampled Point Cloud', 'Position', [300, 300, 800, 600]);
if size(XYZcamSample, 2) > 0
    scatter3(XYZcamSample(1, :), XYZcamSample(2, :), XYZcamSample(3, :), ...
        10, RGBsample', 'filled');
    xlabel('X (meters)'); ylabel('Y (meters)'); zlabel('Z (meters)');
    title(sprintf('Downsampled Point Cloud (view from camera, matching RGB image)'));
    axis equal;
    grid on;
    colorbar;
    % Set camera view: view from behind camera looking along -Z direction
    view(0, -90);  % Azimuth=0 (along X), Elevation=-90 (from above, looking down)
end
fprintf('  [VISUALIZATION] Figure 5: Downsampled point cloud\n');

%% Step 8: Coordinate transformation (camera to world)
fprintf('\nStep 7: Coordinate transformation (camera -> world)...\n');
try
    firstPose = ensureHomogeneousTransform(data.extrinsicsC2W(:, :, 1));
    Rw = firstPose(1:3, 1:3);
    tw = firstPose(1:3, 4);
    
    normalsWorld = normalizeColumns(Rw * normalsCamSample);
    XYZworld = Rw * XYZcamSample + tw;
    
    fprintf('  [OK] Successfully transformed\n');
    fprintf('    Transformation matrix (first 3 rows):\n');
    fprintf('      [%.3f  %.3f  %.3f  %.3f]\n', firstPose(1,1), firstPose(1,2), firstPose(1,3), firstPose(1,4));
    fprintf('      [%.3f  %.3f  %.3f  %.3f]\n', firstPose(2,1), firstPose(2,2), firstPose(2,3), firstPose(2,4));
    fprintf('      [%.3f  %.3f  %.3f  %.3f]\n', firstPose(3,1), firstPose(3,2), firstPose(3,3), firstPose(3,4));
    fprintf('    World coordinate point cloud: %d points\n', size(XYZworld, 2));
    if size(XYZworld, 2) > 0
        fprintf('    X range: %.3f to %.3f meters\n', min(XYZworld(1,:)), max(XYZworld(1,:)));
        fprintf('    Y range: %.3f to %.3f meters\n', min(XYZworld(2,:)), max(XYZworld(2,:)));
        fprintf('    Z range: %.3f to %.3f meters\n', min(XYZworld(3,:)), max(XYZworld(3,:)));
    end
catch ME
    fprintf('  [FAILED] %s\n', ME.message);
    return;
end

% Visualization: Step 7
figure('Name', 'Step 7: Point Cloud in World Coordinates', 'Position', [350, 350, 800, 600]);
if size(XYZworld, 2) > 0
    scatter3(XYZworld(1, :), XYZworld(2, :), XYZworld(3, :), ...
        10, RGBsample', 'filled');
    xlabel('X (meters)'); ylabel('Y (meters)'); zlabel('Z (meters)');
    title('Point Cloud in World Coordinates');
    axis equal;
    grid on;
    colorbar;
    % Set a good viewing angle for world coordinates (3D perspective view)
    view(135, 30);  % Azimuth=135, Elevation=30 (diagonal view from above)
end
fprintf('  [VISUALIZATION] Figure 6: Point cloud in world coordinates\n');

%% Step 9: Final summary visualization
fprintf('\nStep 8: Final summary visualization...\n');
figure('Name', 'Final Summary: Complete Pipeline', 'Position', [400, 400, 1400, 500]);

% Subplot 1: Original RGB image
subplot(1, 4, 1);
imshow(image);
title('1. Original RGB Image');

% Subplot 2: Depth map
subplot(1, 4, 2);
imagesc(depth_raw);
axis image;  % Keep aspect ratio same as original image
axis off;    % Remove axes
colorbar;
title('2. Raw Depth Map');
colormap(gca, 'jet');

% Subplot 3: Point cloud in camera coordinates
subplot(1, 4, 3);
if size(XYZcamSample, 2) > 0
    viz_idx = 1:5:size(XYZcamSample, 2);
    scatter3(XYZcamSample(1, viz_idx), XYZcamSample(2, viz_idx), XYZcamSample(3, viz_idx), ...
        10, RGBsample(:, viz_idx)', 'filled');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('3. Camera Coordinates (camera view)');
    axis equal;
    grid on;
    % Set camera view: view from behind camera looking along -Z direction
    view(0, -90);  % Azimuth=0 (along X), Elevation=-90 (from above, looking down)
end

% Subplot 4: Point cloud in world coordinates
subplot(1, 4, 4);
if size(XYZworld, 2) > 0
    viz_idx = 1:5:size(XYZworld, 2);
    scatter3(XYZworld(1, viz_idx), XYZworld(2, viz_idx), XYZworld(3, viz_idx), ...
        10, RGBsample(:, viz_idx)', 'filled');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('4. World Coordinates (3D view)');
    axis equal;
    grid on;
    % Set a good viewing angle for world coordinates (3D perspective view)
    view(135, 30);  % Azimuth=135, Elevation=30 (diagonal view from above)
end

fprintf('  [VISUALIZATION] Figure 7: Complete pipeline summary\n');

%% Final summary
fprintf('\n=== All Preprocessing Steps Completed Successfully! ===\n');
fprintf('\nFinal Results:\n');
fprintf('  Final point cloud (world): %d points\n', size(XYZworld, 2));
fprintf('  Final colors: %d points\n', size(RGBsample, 2));
fprintf('  Final normals: %d points\n', size(normalsWorld, 2));
fprintf('\nAll visualizations are displayed in separate figures.\n');

%% Helper functions
function idx = selectSampleIndices(numPoints, downsample)
% Select sample indices for downsampling
if numPoints == 0
    idx = zeros(1, 0);
    return;
end
if downsample <= 1
    idx = 1:numPoints;
else
    idx = 1:downsample:numPoints;
end
end

function normals = normalizeColumns(vectors)
% Normalize each column of the input matrix to unit length
norms = sqrt(sum(vectors.^2, 1));
norms(norms < eps) = 1;
normals = vectors ./ norms;
end

function T = ensureHomogeneousTransform(Traw)
% Ensure the transformation matrix is 4x4 homogeneous format
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
