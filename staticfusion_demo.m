% Author: Anqi Wei
% Adapted from StaticFusion
% Reference: StaticFusion: Background Reconstruction for Dense RGB-D SLAM in Dynamic Environments
%            GitHub: https://github.com/raluca-scona/staticfusion

%% STATICFUSION_DEMO
% Incremental static-fusion pipeline inspired by StaticFusion

clear; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
startupPath = fullfile(scriptDir, 'startup.m');
if exist(startupPath, 'file')
    run(startupPath);
else
    error('startup.m is missing.');
end

if exist('visualizeProgress', 'file') ~= 2
    error('visualizeProgress function not found. Please check startup.m includes visualization path.');
end
if exist('visualizeStaticFusionResults', 'file') ~= 2
    error('visualizeStaticFusionResults function not found. Please check startup.m includes visualization path.');
end

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
sequenceName = 'hotel_umd/maryland_hotel3';
frameRange = [];

progress.enableSave = true;
progress.savePath = fullfile(scriptDir, 'staticfusion_progress.mat');
progress.saveEvery = 5;
progress.restoreIfExists = true;

memory.enableMonitoring = true;
memory.warnThreshold = 0.8;
memory.checkEvery = 10;

depthFilterOpts = struct('minDepth', 0.4, 'maxDepth', 4.5, 'medianKernel', 3);
denoiseOpts = struct('radius', 0.04, 'minNeighbors', 12);
normalOpts = struct('K', 25);

processing.downsample = 5;
processing.visualiseEvery = 3;
processing.enableVisualisation = true;

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

cleanupOpts = struct( ...
    'EnableCleanup', true, ...
    'CleanupEvery', 50, ...
    'MinConfidence', 0.5, ...
    'MergeRadius', 0.05, ...
    'NormalSimilarity', 0.9, ...
    'Verbose', true);

meshOpts = struct( ...
    'EnableReconstruction', true, ...
    'Method', 'delaunay', ...
    'MaxDistance', 0.1, ...
    'NormalWeight', 0.8, ...
    'ExportFormat', 'ply', ...
    'ExportPath', fullfile(scriptDir, 'reconstruction_output.ply'), ...
    'IncludeNormals', true, ...
    'IncludeColors', true);

%% Load data and determine frame range
fprintf('[StaticFusion] Loading SUN3D sequence "%s"...\n', sequenceName);

tempData = loadSUN3D(sequenceName, 1);
if isempty(frameRange)
    allData = loadSUN3D(sequenceName);
    totalFrames = length(allData.image);
    frameRange = 1:totalFrames;
    fprintf('[StaticFusion] Processing all %d frames.\n', totalFrames);
    clear allData;
else
    totalFrames = max(frameRange);
    fprintf('[StaticFusion] Processing frames %d to %d (total: %d frames).\n', ...
        min(frameRange), max(frameRange), numel(frameRange));
end

data = loadSUN3D(sequenceName, frameRange);
numFrames = numel(frameRange);

if isempty(data.extrinsicsC2W)
    error('Extrinsic camera poses are unavailable for this sequence.');
end

%% Progress restoration
startFrameIdx = 1;
if progress.restoreIfExists && exist(progress.savePath, 'file')
    try
        fprintf('[StaticFusion] Attempting to restore progress from %s...\n', progress.savePath);
        saved = load(progress.savePath);
        if isfield(saved, 'staticMap') && isfield(saved, 'currentPose') && ...
           isfield(saved, 'idxFrame') && saved.idxFrame < numFrames
            staticMap = saved.staticMap;
            currentPose = saved.currentPose;
            startFrameIdx = saved.idxFrame + 1;
            trajectory = saved.trajectory;
            poses = saved.poses;
            staticCounts = saved.staticCounts;
            dynamicCounts = saved.dynamicCounts;
            lastStaticWorld = saved.lastStaticWorld;
            lastDynamicWorld = saved.lastDynamicWorld;
            lastFrameID = saved.lastFrameID;
            fprintf('[StaticFusion] Restored progress from frame %d. Continuing from frame %d.\n', ...
                saved.idxFrame, startFrameIdx);
        else
            fprintf('[StaticFusion] Saved progress incomplete or already finished. Starting fresh.\n');
            startFrameIdx = 1;
        end
    catch ME
        warning('[StaticFusion] Failed to restore progress: %s. Starting fresh.', ME.message);
        startFrameIdx = 1;
    end
end

if startFrameIdx == 1
    trajectory = zeros(3, numFrames);
    poses = zeros(4, 4, numFrames);
    staticCounts = zeros(1, numFrames);
    dynamicCounts = zeros(1, numFrames);
    lastStaticWorld = zeros(3, 0);
    lastDynamicWorld = zeros(3, 0);
    lastFrameID = frameRange(1);
end

%% Initialise with first frame
if startFrameIdx == 1
    fprintf('[StaticFusion] Initialising map with frame %d...\n', frameRange(1));
    frameData = loadFrameData(data, frameRange, 1, depthFilterOpts, denoiseOpts, normalOpts);

    firstPose = ensureHomogeneousTransform(data.extrinsicsC2W(:, :, frameData.frameID));
    poses(:, :, 1) = firstPose;
    trajectory(:, 1) = firstPose(1:3, 4);

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
    currentPose = firstPose;
else
    fprintf('[StaticFusion] Using restored state. Starting from frame %d.\n', startFrameIdx);
end

%% Main processing loop
for idxFrame = startFrameIdx:numFrames
    try
        frameID = frameRange(idxFrame);
        fprintf('[StaticFusion] Processing frame %d (%d/%d)...\n', frameID, idxFrame, numFrames);

        fprintf('  Loading frame data...\n');
        frameData = loadFrameData(data, frameRange, idxFrame, depthFilterOpts, denoiseOpts, normalOpts);
        
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
        trackArgs = structToNameValue(trackingOpts);
        trackRes = trackAgainstStaticMap(staticMap, currentPose, pointsCam, segRes.staticMask, trackArgs{:});

        if trackRes.success
            currentPose = trackRes.Twc;
            fprintf('  Pose tracking succeeded (RMSE: %.4f)\n', trackRes.rmse);
        else
            warning('Pose tracking failed for frame %d; keeping previous pose.', frameID);
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

        if processing.enableVisualisation && mod(idxFrame - 1, processing.visualiseEvery) == 0
            try
                visualizeProgress(staticMap, currentPose, pointsCam, segRes.staticMask, segRes.dynamicMask, trajectory(:, 1:idxFrame));
            catch ME
                warning('Visualization failed at frame %d: %s', frameID, ME.message);
            end
        end
        fprintf('  Frame %d complete.\n\n', frameID);
        
        if cleanupOpts.EnableCleanup && mod(idxFrame, cleanupOpts.CleanupEvery) == 0
            try
                fprintf('[StaticFusion] Running periodic map cleanup...\n');
                cleanupArgs = {};
                if isfield(cleanupOpts, 'MinConfidence')
                    cleanupArgs{end+1} = 'MinConfidence';
                    cleanupArgs{end+1} = cleanupOpts.MinConfidence;
                end
                if isfield(cleanupOpts, 'MergeRadius')
                    cleanupArgs{end+1} = 'MergeRadius';
                    cleanupArgs{end+1} = cleanupOpts.MergeRadius;
                end
                if isfield(cleanupOpts, 'NormalSimilarity')
                    cleanupArgs{end+1} = 'NormalSimilarity';
                    cleanupArgs{end+1} = cleanupOpts.NormalSimilarity;
                end
                if isfield(cleanupOpts, 'Verbose')
                    cleanupArgs{end+1} = 'Verbose';
                    cleanupArgs{end+1} = cleanupOpts.Verbose;
                end
                staticMap = cleanupStaticMap(staticMap, cleanupArgs{:});
                fprintf('[StaticFusion] Map cleanup complete. Current surfels: %d\n', ...
                    size(staticMap.positions, 2));
            catch ME
                warning('[StaticFusion] Map cleanup failed: %s', ME.message);
            end
        end
        
        if memory.enableMonitoring && mod(idxFrame, memory.checkEvery) == 0
            if ispc
                [~, memInfo] = memory;
                memUsage = memInfo.MemUsedMATLAB / memInfo.MemAvailableAllArrays;
            else
                [~, memStr] = unix('vm_stat | grep "Pages free"');
                memUsage = 0.5;
            end
            if memUsage > memory.warnThreshold
                warning('[StaticFusion] High memory usage detected (%.1f%%). Consider reducing downsample or enabling map cleanup.', ...
                    memUsage * 100);
            end
        end
        
        if progress.enableSave && mod(idxFrame, progress.saveEvery) == 0
            try
                fprintf('[StaticFusion] Saving progress at frame %d...\n', frameID);
                save(progress.savePath, 'staticMap', 'currentPose', 'trajectory', 'poses', ...
                    'staticCounts', 'dynamicCounts', 'lastStaticWorld', 'lastDynamicWorld', ...
                    'lastFrameID', 'idxFrame', 'frameRange', '-v7.3');
                fprintf('[StaticFusion] Progress saved.\n');
            catch ME
                warning('[StaticFusion] Failed to save progress: %s', ME.message);
            end
        end
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

if progress.enableSave
    try
        fprintf('[StaticFusion] Saving final progress...\n');
        save(progress.savePath, 'staticMap', 'currentPose', 'trajectory', 'poses', ...
            'staticCounts', 'dynamicCounts', 'lastStaticWorld', 'lastDynamicWorld', ...
            'lastFrameID', 'idxFrame', 'frameRange', '-v7.3');
        fprintf('[StaticFusion] Final progress saved.\n');
    catch ME
        warning('[StaticFusion] Failed to save final progress: %s', ME.message);
    end
end

%% Final visualization
if processing.enableVisualisation
    fprintf('[StaticFusion] Generating final visualizations...\n');
    fprintf('[StaticFusion] Data summary:\n');
    fprintf('  Static map surfels: %d\n', size(staticMap.positions, 2));
    fprintf('  Trajectory points: %d\n', size(trajectory, 2));
    fprintf('  Frame range: %d to %d\n', frameRange(1), frameRange(end));

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

%% Mesh reconstruction and export
if meshOpts.EnableReconstruction
    fprintf('[StaticFusion] Starting mesh reconstruction...\n');
    try
        meshArgs = {};
        meshArgs{end+1} = 'Method';
        meshArgs{end+1} = meshOpts.Method;
        meshArgs{end+1} = 'MaxDistance';
        meshArgs{end+1} = meshOpts.MaxDistance;
        meshArgs{end+1} = 'NormalWeight';
        meshArgs{end+1} = meshOpts.NormalWeight;
        meshArgs{end+1} = 'Verbose';
        meshArgs{end+1} = true;
        
        mesh = surfelToMesh(staticMap, meshArgs{:});
        
        fprintf('[StaticFusion] Mesh reconstruction complete.\n');
        fprintf('[StaticFusion] Exporting mesh to %s...\n', meshOpts.ExportPath);
        
        exportArgs = {};
        exportArgs{end+1} = 'Format';
        exportArgs{end+1} = meshOpts.ExportFormat;
        exportArgs{end+1} = 'IncludeNormals';
        exportArgs{end+1} = meshOpts.IncludeNormals;
        exportArgs{end+1} = 'IncludeColors';
        exportArgs{end+1} = meshOpts.IncludeColors;
        
        exportMesh(mesh, meshOpts.ExportPath, exportArgs{:});
        fprintf('[StaticFusion] Mesh export complete.\n');
    catch ME
        fprintf('[StaticFusion] ERROR: Mesh reconstruction/export failed: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('  Error location: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
        end
    end
else
    fprintf('[StaticFusion] Mesh reconstruction disabled.\n');
end

%% Helper functions
function frameData = loadFrameData(data, frameRange, localIdx, depthFilterOpts, denoiseOpts, normalOpts)
    frameID = frameRange(localIdx);
    image = imread(data.image{localIdx});
    depth = double(depthRead(data.depth{localIdx}));

    [XYZcam, RGB] = depthToPointCloud(image, depth, data.K, 'depthFilter', depthFilterOpts);
    
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
catch ME
    warning('loadFrameData:Error', 'Error loading frame %d: %s', frameRange(localIdx), ME.message);
    frameData = struct('frameID', frameRange(localIdx), 'XYZcam', zeros(3,0), 'RGB', zeros(3,0), 'normalsCam', zeros(3,0));
end
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
