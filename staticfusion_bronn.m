%% STATICFUSION - Bronn Dataset
% Demonstrates an incremental static-fusion pipeline inspired by StaticFusion.
% The script loads a Bronn sequence, builds a surfel-based static background
% map, segments static/dynamic observations each frame, tracks the camera pose
% against the map, and visualises the reconstruction progress.

clear; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
startupPath = fullfile(scriptDir, 'startup.m');
if exist(startupPath, 'file')
    run(startupPath);
else
    error('startup.m is missing.');
end

% Verify visualization functions are available
if exist('visualizeProgress', 'file') ~= 2
    error('visualizeProgress function not found. Please check startup.m includes visualization path.');
end
if exist('visualizeStaticFusionResults', 'file') ~= 2
    error('visualizeStaticFusionResults function not found. Please check startup.m includes visualization path.');
end

% Check for required functions
if ~exist('knnsearch', 'builtin') && ~exist('knnsearch', 'file')
    error('knnsearch function not found. Please ensure Statistics and Machine Learning Toolbox is installed.');
end
fprintf('[StaticFusion] Checking dependencies...\n');
fprintf('  knnsearch: %s\n', func2str(@knnsearch));
fprintf('  initializeStaticMap: %s\n', func2str(@initializeStaticMap));
fprintf('  predictStaticMap: %s\n', func2str(@predictStaticMap));
fprintf('  updateStaticMap: %s\n', func2str(@updateStaticMap));
fprintf('  estimateStaticMask: %s\n', func2str(@estimateStaticMask));
fprintf('  trackAgainstStaticMap: %s\n', func2str(@trackAgainstStaticMap));
fprintf('Dependencies OK.\n\n');

%% User configuration
bronn_datadir = "D:\Bronn";
sequenceName = "rgbd_bonn_crowd";  % Dataset folder name (kept as bonn for compatibility)
base_sequence_dir = fullfile(bronn_datadir,sequenceName);

frameRange = 189:286;  % frames to process incrementally

depthFilterOpts = struct('minDepth', 0.4, 'maxDepth', 4.5, 'medianKernel', 3);
denoiseOpts = struct('radius', 0.04, 'minNeighbors', 12);
normalOpts = struct('K', 25);

processing.downsample = 5;      % only keep every Nth point for processing
processing.visualiseEvery = 1;  % update visualisation every N frames
processing.enableVisualisation = true;  % set to true to show figures

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

trackingOpts = struct( ...
    'MaxIterations', 12, ...
    'MaxCorrespondence', 0.08, ...
    'NormalSimilarity', cosd(50), ...
    'Damping', 1e-3, ...
    'MinInliers', 120, ...
    'Verbose', false);

% ===== Configuration flags =====
% Control whether to use ground truth for initialization and debugging
USE_GT_INIT = true;      % Use ground truth for first frame initialization
USE_GT_FALLBACK = false; % Use ground truth when tracking fails (for debugging)
ENABLE_EVALUATION = true; % Enable evaluation against ground truth

%% Load data
fprintf('[StaticFusion] Loading Bronn sequence list "%s"...\n', sequenceName);

data.basedir = base_sequence_dir;
fid = fopen(fullfile(base_sequence_dir,'rgb.txt'));
if fid == -1
    error('Cannot open rgb.txt file. Check path: %s', fullfile(base_sequence_dir,'rgb.txt'));
end
temp = textscan(fid,'%f %s','HeaderLines',2);
fclose(fid);
data.rgblist = table(temp{1},string(temp{2}),'VariableNames',{'time_posix','filename'});

fid = fopen(fullfile(base_sequence_dir,'depth.txt'));
if fid == -1
    error('Cannot open depth.txt file. Check path: %s', fullfile(base_sequence_dir,'depth.txt'));
end
temp = textscan(fid,'%f %s','HeaderLines',2);
fclose(fid);
data.depthlist = table(temp{1},string(temp{2}),'VariableNames',{'time_posix','filename'});

fid = fopen(fullfile(base_sequence_dir,"groundtruth.txt"));
if fid == -1
    error('Cannot open groundtruth.txt file. Check path: %s', fullfile(base_sequence_dir,"groundtruth.txt"));
end
% timestamp tx ty tz qx qy qz qw
temp = textscan(fid,"%f %f %f %f %f %f %f %f",'HeaderLines',2);
fclose(fid);
data.poseTruth = table(temp{1},[temp{2:4}],[temp{8},temp{5:7}],'VariableNames',{'timestamp','txyz','quat'});
clear temp;

numFrames = numel(frameRange);

if isempty(data.poseTruth)
    error('Extrinsic camera poses are unavailable for this sequence.');
end

%% Containers for outputs
trajectory = zeros(3, numFrames);
poses = zeros(4, 4, numFrames);
staticCounts = zeros(1, numFrames);
dynamicCounts = zeros(1, numFrames);
lastStaticWorld = zeros(3, 0);
lastDynamicWorld = zeros(3, 0);
lastFrameID = frameRange(1);

% ===== Store ground truth poses for evaluation =====
if ENABLE_EVALUATION
    gtPoses = zeros(4, 4, numFrames);
    gtTrajectory = zeros(3, numFrames);
end

%% Initialise with first frame
fprintf('[StaticFusion] Initialising map with frame %d...\n', frameRange(1));
frameData = loadFrameDataBronn(data, frameRange, 1, depthFilterOpts, denoiseOpts, normalOpts);

[~,poseIdx] = min(abs(data.rgblist.time_posix(frameRange(1)) - data.poseTruth.timestamp));

% ===== FIXED: Proper initial pose =====
if USE_GT_INIT
    try
        firstPose = ensureHomogeneousTransform(bronnTransform(data.poseTruth(poseIdx,:)));
        fprintf('  Using ground truth pose for initialization.\n');
    catch ME
        warning('Failed to load ground truth pose, using identity: %s', ME.message);
        firstPose = eye(4);
    end
else
    firstPose = eye(4);  % Use identity matrix instead of broken [eye(4,3) zeros(4,1)]
    fprintf('  Using identity pose for initialization.\n');
end

poses(:, :, 1) = firstPose;
trajectory(:, 1) = firstPose(1:3, 4);

% Store ground truth for first frame
if ENABLE_EVALUATION
    gtPoses(:, :, 1) = ensureHomogeneousTransform(bronnTransform(data.poseTruth(poseIdx,:)));
    gtTrajectory(:, 1) = gtPoses(1:3, 4, 1);
end

sampleIdx = selectSampleIndices(size(frameData.XYZcam, 2), processing.downsample);

XYZcamSample = frameData.XYZcam(:, sampleIdx);
RGBsample = frameData.RGB(:, sampleIdx);
normalsCamSample = frameData.normalsCam(:, sampleIdx);

Rw = firstPose(1:3, 1:3);
normalsWorld = normalizeColumns(Rw * normalsCamSample);
XYZworld = Rw * XYZcamSample + firstPose(1:3, 4);

pcInit = struct('XYZ', XYZworld, 'RGB', RGBsample);
mapInitArgs = structToNameValue(mapInitOpts);
staticMap = initializeStaticMap(pcInit, normalsWorld, mapInitArgs{:});
lastStaticWorld = XYZworld;
lastDynamicWorld = zeros(3, 0);
lastFrameID = frameRange(1);

staticCounts(1) = size(XYZcamSample, 2);
dynamicCounts(1) = 0;

%% Main processing loop
currentPose = firstPose;
for idxFrame = 2:numFrames
    try
        frameID = frameRange(idxFrame);
        fprintf('[StaticFusion] Processing frame %d (%d/%d)...\n', frameID, idxFrame, numFrames);

        fprintf('  Loading frame data...\n');
        frameData = loadFrameDataBronn(data, frameRange, idxFrame, depthFilterOpts, denoiseOpts, normalOpts);
        
        if isempty(frameData.XYZcam) || size(frameData.XYZcam, 2) == 0
            warning('Frame %d has no points, skipping.', frameID);
            continue;
        end
        
        sampleIdx = selectSampleIndices(size(frameData.XYZcam, 2), processing.downsample);
        
        if isempty(sampleIdx)
            warning('Frame %d has no sampled points, skipping.', frameID);
            continue;
        end

        pointsCam = frameData.XYZcam(:, sampleIdx);
        colours = frameData.RGB(:, sampleIdx);
        normalsCam = frameData.normalsCam(:, sampleIdx);
        
        % Validate sizes
        if size(pointsCam, 2) ~= size(colours, 2) || size(pointsCam, 2) ~= size(normalsCam, 2)
            error('Size mismatch: points=%d, colours=%d, normals=%d', ...
                size(pointsCam, 2), size(colours, 2), size(normalsCam, 2));
        end
        
        fprintf('  Points: %d, Colours: %d, Normals: %d\n', ...
            size(pointsCam, 2), size(colours, 2), size(normalsCam, 2));

        fprintf('  Predicting static map...\n');
        prediction = predictStaticMap(staticMap, currentPose);

        fprintf('  Estimating static mask (first pass)...\n');
        segArgs = [structToNameValue(segmentationOpts), {'NormalsCam', normalsCam}];
        segRes = estimateStaticMask(pointsCam, colours, prediction, segArgs{:});

        fprintf('  Tracking pose...\n');
        % ===== Suppress verbose warnings =====
        warning('off', 'trackAgainstStaticMap:*');
        trackArgs = structToNameValue(trackingOpts);
        trackRes = trackAgainstStaticMap(staticMap, currentPose, pointsCam, segRes.staticMask, trackArgs{:});
        warning('on', 'trackAgainstStaticMap:*');

        if trackRes.success
            currentPose = trackRes.Twc;
            fprintf('  Pose tracking succeeded (RMSE: %.4f)\n', trackRes.rmse);
        else
            warning('Pose tracking failed for frame %d; keeping previous pose.', frameID);
            % ===== Optional ground truth fallback for debugging =====
            if USE_GT_FALLBACK
                [~,dbposeIdx] = min(abs(data.rgblist.time_posix(frameID) - data.poseTruth.timestamp));
                currentPose = ensureHomogeneousTransform(bronnTransform(data.poseTruth(dbposeIdx,:)));
                fprintf('  Using ground truth pose as fallback.\n');
            end
        end

        fprintf('  Re-predicting static map...\n');
        prediction = predictStaticMap(staticMap, currentPose);

        fprintf('  Estimating static mask (second pass)...\n');
        segArgs = [structToNameValue(segmentationOpts), ...
            {'NormalsCam', normalsCam, 'PriorDynamicMask', segRes.dynamicMask}];
        segRes = estimateStaticMask(pointsCam, colours, prediction, segArgs{:});

        staticPoints = pointsCam(:, segRes.staticMask);
        staticNormalsCam = normalsCam(:, segRes.staticMask);
        staticColours = colours(:, segRes.staticMask);
        dynamicPoints = pointsCam(:, segRes.dynamicMask);
        
        Rw = currentPose(1:3, 1:3);
        tw = currentPose(1:3, 4);
        lastStaticWorld = Rw * staticPoints + tw;
        lastDynamicWorld = Rw * dynamicPoints + tw;
        lastFrameID = frameID;

        fprintf('  Updating static map (%d static points)...\n', size(staticPoints, 2));
        if ~isempty(staticPoints)
            staticMap = updateStaticMap(staticMap, currentPose, staticPoints, staticNormalsCam, staticColours);
        end

        dynamicCounts(idxFrame) = nnz(segRes.dynamicMask);
        staticCounts(idxFrame) = nnz(segRes.staticMask);
        poses(:, :, idxFrame) = currentPose;
        trajectory(:, idxFrame) = currentPose(1:3, 4);
        
        % ===== Store ground truth for evaluation =====
        if ENABLE_EVALUATION
            [~,gtPoseIdx] = min(abs(data.rgblist.time_posix(frameID) - data.poseTruth.timestamp));
            gtPoses(:, :, idxFrame) = ensureHomogeneousTransform(bronnTransform(data.poseTruth(gtPoseIdx,:)));
            gtTrajectory(:, idxFrame) = gtPoses(1:3, 4, idxFrame);
        end

        % Visualization during processing
        if processing.enableVisualisation && mod(idxFrame - 1, processing.visualiseEvery) == 0
            try
                visualizeProgress(staticMap, currentPose, pointsCam, segRes.staticMask, segRes.dynamicMask, trajectory(:, 1:idxFrame));
            catch ME
                warning('Visualization failed at frame %d: %s', frameID, ME.message);
            end
        end
        fprintf('  Frame %d complete.\n\n', frameID);
    catch ME
        fprintf('\n!!! ERROR at frame %d (index %d) !!!\n', frameRange(idxFrame), idxFrame);
        fprintf('Error message: %s\n', ME.message);
        fprintf('Error location: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
        fprintf('Stack trace:\n');
        for k = 1:min(5, numel(ME.stack))
            fprintf('  %s at line %d\n', ME.stack(k).name, ME.stack(k).line);
        end
        rethrow(ME);
    end
end

fprintf('[StaticFusion] Processing complete. Final map contains %d surfels.\n', size(staticMap.positions, 2));

%% Final visualization
if processing.enableVisualisation
    fprintf('[StaticFusion] Generating final visualizations...\n');
    fprintf('[StaticFusion] Data summary:\n');
    fprintf('  Static map surfels: %d\n', size(staticMap.positions, 2));
    fprintf('  Trajectory points: %d\n', size(trajectory, 2));
    fprintf('  Frame range: %d to %d\n', frameRange(1), frameRange(end));

    % Only visualize if we have valid data
    if ~isempty(staticMap.positions) && size(staticMap.positions, 2) > 0 && ...
       ~isempty(trajectory) && size(trajectory, 2) >= 2
        try
            visualizeStaticFusionResults(staticMap, poses, trajectory, staticCounts, dynamicCounts, frameRange, ...
                'LastStaticWorld', lastStaticWorld, 'LastDynamicWorld', lastDynamicWorld, 'LastFrameID', lastFrameID);
            fprintf('[StaticFusion] Final visualization completed successfully.\n');
        catch ME
            fprintf('[StaticFusion] ERROR: Final visualization failed: %s\n', ME.message);
            if ~isempty(ME.stack)
                fprintf('  Error location: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
            end
        end
    else
        fprintf('[StaticFusion] Skipping visualization: insufficient data.\n');
        fprintf('  Map points: %d, Trajectory points: %d\n', ...
            size(staticMap.positions, 2), size(trajectory, 2));
    end
else
    fprintf('[StaticFusion] Visualisation disabled (processing.enableVisualisation = false).\n');
end

%% Evaluation section
if ENABLE_EVALUATION
    fprintf('\n[StaticFusion] Evaluating against ground truth...\n');
    evaluateTrajectory(poses, gtPoses, trajectory, gtTrajectory, frameRange);
end

%% Helper functions
function frameData = loadFrameDataBronn(data, frameRange, localIdx, depthFilterOpts, denoiseOpts, normalOpts)
    frameID = frameRange(localIdx);

    image = imread(fullfile(data.basedir,data.rgblist.filename(frameID)));

    [~,depthIdx] = min(abs(data.rgblist.time_posix(frameID) - data.depthlist.time_posix));
    depth = depthReadTUM(fullfile(data.basedir,data.depthlist.filename(depthIdx)));

    [XYZcam, RGB] = depthToPointCloudBronn(image, depth, 'depthFilter', depthFilterOpts);
    
    if isempty(XYZcam) || size(XYZcam, 2) == 0
        warning('loadFrameData:EmptyCloud', 'Frame %d produced empty point cloud.', frameID);
        frameData = struct('frameID', frameID, 'XYZcam', zeros(3,0), 'RGB', zeros(3,0), 'normalsCam', zeros(3,0));
        return;
    end

    if ~isempty(denoiseOpts)
        denoiseArgs = structToNameValue(denoiseOpts);
        [XYZcam, RGB] = denoisePointCloud(XYZcam, RGB, denoiseArgs{:});
    end
    
    if isempty(XYZcam) || size(XYZcam, 2) == 0
        warning('loadFrameData:EmptyAfterDenoise', 'Frame %d point cloud empty after denoising.', frameID);
        frameData = struct('frameID', frameID, 'XYZcam', zeros(3,0), 'RGB', zeros(3,0), 'normalsCam', zeros(3,0));
        return;
    end

    normalArgs = structToNameValue(normalOpts);
    normalsCam = estimateNormals(XYZcam, normalArgs{:}, 'ViewPoint', [0; 0; 0]);
    
    if size(normalsCam, 2) ~= size(XYZcam, 2)
        warning('loadFrameData:NormalMismatch', 'Normal count mismatch, using default normals.');
        normalsCam = zeros(3, size(XYZcam, 2));
        normalsCam(3, :) = 1;
    end

    frameData = struct('frameID', frameID, 'XYZcam', XYZcam, 'RGB', RGB, 'normalsCam', normalsCam);
end

function idx = selectSampleIndices(numPoints, downsample)
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

function args = structToNameValue(s)
if isempty(s)
    args = {};
    return;
end
fields = fieldnames(s);
args = cell(1, numel(fields) * 2);
for idx = 1:numel(fields)
    args{2*idx - 1} = fields{idx};
    args{2*idx} = s.(fields{idx});
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

function evaluateTrajectory(poses, gtPoses, trajectory, gtTrajectory, frameRange)
    % Evaluate trajectory accuracy against ground truth
    
    % Compute Absolute Trajectory Error (ATE)
    numFrames = size(poses, 3);
    ate_translation = zeros(1, numFrames);
    ate_rotation = zeros(1, numFrames);
    
    for i = 1:numFrames
        % Translation error
        t_est = poses(1:3, 4, i);
        t_gt = gtPoses(1:3, 4, i);
        ate_translation(i) = norm(t_est - t_gt);
        
        % Rotation error (angle in degrees)
        R_est = poses(1:3, 1:3, i);
        R_gt = gtPoses(1:3, 1:3, i);
        R_diff = R_est * R_gt';
        trace_R = trace(R_diff);
        trace_R = max(-1, min(1, trace_R));  % Clamp to [-1, 1]
        angle_rad = acos((trace_R - 1) / 2);
        ate_rotation(i) = rad2deg(angle_rad);
    end
    
    fprintf('  Average Translation Error: %.4f m\n', mean(ate_translation));
    fprintf('  Average Rotation Error: %.4f deg\n', mean(ate_rotation));
    fprintf('  Max Translation Error: %.4f m\n', max(ate_translation));
    fprintf('  Max Rotation Error: %.4f deg\n', max(ate_rotation));
    fprintf('  RMSE Translation Error: %.4f m\n', sqrt(mean(ate_translation.^2)));
    
    % Visualize trajectory comparison
    figure('Name', 'Trajectory Evaluation');
    
    subplot(2, 2, 1);
    plot(1:numFrames, ate_translation, 'b-', 'LineWidth', 2);
    xlabel('Frame');
    ylabel('Translation Error (m)');
    title('Absolute Translation Error');
    grid on;
    
    subplot(2, 2, 2);
    plot(1:numFrames, ate_rotation, 'r-', 'LineWidth', 2);
    xlabel('Frame');
    ylabel('Rotation Error (deg)');
    title('Absolute Rotation Error');
    grid on;
    
    subplot(2, 2, 3);
    plot3(trajectory(1, :), trajectory(2, :), trajectory(3, :), 'b-', 'LineWidth', 2);
    hold on;
    plot3(gtTrajectory(1, :), gtTrajectory(2, :), gtTrajectory(3, :), 'r--', 'LineWidth', 2);
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('Trajectory Comparison');
    legend('Estimated', 'Ground Truth', 'Location', 'best');
    grid on;
    axis equal;
    
    subplot(2, 2, 4);
    plot(frameRange, ate_translation, 'b-', 'LineWidth', 2);
    hold on;
    plot(frameRange, ate_rotation / 10, 'r-', 'LineWidth', 2);  % Scale rotation for visibility
    xlabel('Frame ID');
    ylabel('Error');
    title('Error Over Time');
    legend('Translation (m)', 'Rotation (deg/10)', 'Location', 'best');
    grid on;
end
