% Author: Anqi Wei
% Adapted from StaticFusion
% Reference: StaticFusion: Background Reconstruction for Dense RGB-D SLAM in Dynamic Environments
%            GitHub: https://github.com/raluca-scona/staticfusion

%% STATICFUSION - Bonn Dataset
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
sequenceName = "rgbd_bonn_crowd";
base_sequence_dir = fullfile(scriptDir, sequenceName);

fprintf('[StaticFusion] Dataset path: %s\n', base_sequence_dir);

frameRange = 189:286;

depthFilterOpts = struct('minDepth', 0.3, 'maxDepth', 5.0, 'medianKernel', 5);
denoiseOpts = struct('radius', 0.05, 'minNeighbors', 8);
normalOpts = struct('K', 30);

processing.downsample = 3;
processing.visualiseEvery = 1;
processing.enableVisualisation = true;

mapInitOpts = struct( ...
    'DefaultConfidence', 2.0, ...
    'DefaultRadius', 0.03, ...
    'MergeRadius', 0.04, ...
    'ObservationWeight', 1.0, ...
    'MaxConfidence', 8.0, ...
    'ConfidenceDecay', 0.04);

segmentationOpts = struct( ...
    'GeometryThreshold', 0.06, ...
    'ColorThreshold', 0.20, ...
    'NormalThreshold', pi/4, ...
    'EnergyThreshold', 2.5, ...
    'EnergyWeights', struct('geometry', 1.2, 'colour', 1.0, 'normal', 0.7, 'prior', 1.0), ...
    'MinConfidence', 0.2);

trackingOpts = struct( ...
    'MaxIterations', 20, ...
    'MaxCorrespondence', 0.12, ...
    'NormalSimilarity', cosd(60), ...
    'Damping', 1e-3, ...
    'MinInliers', 40, ...
    'Verbose', false);

USE_GT_INIT = false;
USE_GT_FALLBACK = false;
USE_GT_ONLY = false;
USE_FRAME_TO_FRAME = true;
ENABLE_EVALUATION = true;
ENABLE_POSE_DIAGNOSTICS = true;

%% Load data
fprintf('[StaticFusion] Loading Bonn sequence "%s"...\n', sequenceName);

data = loadBonn(base_sequence_dir, frameRange);
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

if ENABLE_EVALUATION
    gtPoses = zeros(4, 4, numFrames);
    gtTrajectory = zeros(3, numFrames);
end

%% Initialise with first frame
fprintf('[StaticFusion] Initialising map with frame %d...\n', frameRange(1));
frameData = loadFrameDataBonn(data, frameRange, 1, depthFilterOpts, denoiseOpts, normalOpts);

[~,poseIdx] = min(abs(data.rgblist.time_posix(frameRange(1)) - data.poseTruth.timestamp));

if USE_GT_INIT
    try
        % TUM RGB-D format: timestamp tx ty tz qx qy qz qw
        % Reference: TUM RGB-D dataset format standard
        poseRow = data.poseTruth(poseIdx, :);
        txyz = poseRow.txyz;
        quat = poseRow.quat;
        quat_tum = [quat(2), quat(3), quat(4), quat(1)];
        firstPose = tumPoseToTransform(txyz, quat_tum);
        
        if ENABLE_POSE_DIAGNOSTICS
            fprintf('  Using TUM format pose directly.\n');
            fprintf('  Twc translation: [%.4f, %.4f, %.4f]\n', ...
                firstPose(1,4), firstPose(2,4), firstPose(3,4));
            fprintf('  GT translation: [%.4f, %.4f, %.4f]\n', ...
                txyz(1), txyz(2), txyz(3));
            fprintf('  Rotation determinant: %.6f\n', det(firstPose(1:3, 1:3)));
        end
        fprintf('  Using ground truth pose for initialization (TUM format).\n');
    catch ME
        warning('Failed to load ground truth pose, using identity: %s', ME.message);
        firstPose = eye(4);
    end
else
    firstPose = eye(4);
    fprintf('  Using identity pose for initialization.\n');
end

poses(:, :, 1) = firstPose;
trajectory(:, 1) = firstPose(1:3, 4);

if ENABLE_EVALUATION
    poseRow_gt = data.poseTruth(poseIdx, :);
    txyz_gt = poseRow_gt.txyz;
    quat_gt = poseRow_gt.quat;
    quat_tum_gt = [quat_gt(2), quat_gt(3), quat_gt(4), quat_gt(1)];  % [qx, qy, qz, qw]
    gtPoses(:, :, 1) = tumPoseToTransform(txyz_gt, quat_tum_gt);
    gtTrajectory(:, 1) = gtPoses(1:3, 4, 1);
end

sampleIdx = selectSampleIndices(size(frameData.XYZcam, 2), processing.downsample);

XYZcamSample = frameData.XYZcam(:, sampleIdx);
RGBsample = frameData.RGB(:, sampleIdx);
normalsCamSample = frameData.normalsCam(:, sampleIdx);

Rw = firstPose(1:3, 1:3);
tw = firstPose(1:3, 4);
normalsWorld = normalizeColumns(Rw * normalsCamSample);
XYZworld = Rw * XYZcamSample + tw;

if ENABLE_POSE_DIAGNOSTICS
    rotDet = det(Rw);
    if abs(rotDet - 1.0) > 0.01
        warning('Initial pose rotation matrix determinant is %.6f', rotDet);
    end
    shouldBeIdentity = Rw * Rw' - eye(3);
    if any(abs(shouldBeIdentity(:)) > 0.01)
        warning('Initial pose rotation matrix is not orthogonal (max error: %.6f)', max(abs(shouldBeIdentity(:))));
    end
    fprintf('  Initial pose validation:\n');
    fprintf('    Translation: [%.4f, %.4f, %.4f]\n', tw(1), tw(2), tw(3));
    fprintf('    Rotation det: %.6f\n', rotDet);
    fprintf('    Points transformed: %d\n', size(XYZworld, 2));
end

pcInit = struct('XYZ', XYZworld, 'RGB', RGBsample);
mapInitArgs = structToNameValue(mapInitOpts);
staticMap = initializeStaticMap(pcInit, normalsWorld, mapInitArgs{:});
lastStaticWorld = XYZworld;
lastDynamicWorld = zeros(3, 0);
lastFrameID = frameRange(1);

staticCounts(1) = size(XYZcamSample, 2);
dynamicCounts(1) = 0;

prevPointsWorld = [];
prevNormalsWorld = [];

%% Main processing loop
currentPose = firstPose;
for idxFrame = 2:numFrames
    try
        frameID = frameRange(idxFrame);
        fprintf('[StaticFusion] Processing frame %d (%d/%d)...\n', frameID, idxFrame, numFrames);

        fprintf('  Loading frame data...\n');
        frameData = loadFrameDataBonn(data, frameRange, idxFrame, depthFilterOpts, denoiseOpts, normalOpts);
        
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

        if USE_GT_ONLY
            [~,gtPoseIdx] = min(abs(data.rgblist.time_posix(frameID) - data.poseTruth.timestamp));
            poseRow_gt = data.poseTruth(gtPoseIdx, :);
            txyz_gt = poseRow_gt.txyz;
            quat_gt = poseRow_gt.quat;
            quat_tum_gt = [quat_gt(2), quat_gt(3), quat_gt(4), quat_gt(1)];
            currentPose = tumPoseToTransform(txyz_gt, quat_tum_gt);
            
            rotDet = det(currentPose(1:3, 1:3));
            if abs(rotDet - 1.0) > 0.01
                error('GT pose rotation matrix invalid: det=%.6f', rotDet);
            end
            
            fprintf('  [TEST MODE] Using ground truth pose.\n');
            fprintf('  GT pose translation: [%.4f, %.4f, %.4f]\n', txyz_gt(1), txyz_gt(2), txyz_gt(3));
            trackRes.success = true;
            trackRes.rmse = 0;
        else
            fprintf('  Tracking pose...\n');
            
            trackRes = struct();
            trackRes.success = false;
            
            if USE_FRAME_TO_FRAME && ~isempty(prevPointsWorld) && size(prevPointsWorld, 2) > 100
                fprintf('  Trying frame-to-frame tracking...\n');
                
                staticPointsCam = pointsCam(:, segRes.staticMask);
                staticNormalsCam = normalsCam(:, segRes.staticMask);
                
                if size(staticPointsCam, 2) > 50
                    initialPoseForTracking = currentPose;
                    if idxFrame > 2
                        prevPose = poses(:, :, idxFrame - 1);
                        prevPrevPose = poses(:, :, idxFrame - 2);
                        relMotion = inv(prevPrevPose) * prevPose;
                        relTransNorm = norm(relMotion(1:3, 4));
                        if relTransNorm > 0.01 && relTransNorm < 0.3
                            predictedPose = prevPose * relMotion;
                            rotDet = det(predictedPose(1:3, 1:3));
                            if abs(rotDet - 1.0) < 0.01
                                initialPoseForTracking = predictedPose;
                            end
                        end
                    end
                    
                    frameTrackOpts = trackingOpts;
                    if isfield(frameTrackOpts, 'Verbose')
                        frameTrackOpts = rmfield(frameTrackOpts, 'Verbose');
                    end
                    frameTrackArgs = structToNameValue(frameTrackOpts);
                    trackRes = trackFrameToFrame(prevPointsWorld, prevNormalsWorld, ...
                        staticPointsCam, staticNormalsCam, initialPoseForTracking, frameTrackArgs{:});
                    
                    if trackRes.success
                        fprintf('  Frame-to-frame tracking succeeded (RMSE: %.4f, inliers: %d)\n', ...
                            trackRes.rmse, trackRes.numInliers);
                        currentPose = trackRes.Twc;
                    else
                        fprintf('  Frame-to-frame tracking failed, trying static map tracking...\n');
                    end
                end
            end
            
            if ~trackRes.success
                if isempty(staticMap.positions) || size(staticMap.positions, 2) < 100
                    warning('Static map is too small (%d points)', size(staticMap.positions, 2));
                end
                
                numStaticPoints = nnz(segRes.staticMask);
                fprintf('  Static points for tracking: %d / %d (%.1f%%)\n', ...
                    numStaticPoints, size(pointsCam, 2), 100 * numStaticPoints / size(pointsCam, 2));
                
                initialPoseForTracking = currentPose;
                if trackRes.success
                    initialPoseForTracking = trackRes.Twc;
                    fprintf('  Using frame-to-frame result as initial estimate for static map tracking.\n');
                elseif idxFrame > 2
                    prevPose = poses(:, :, idxFrame - 1);
                    prevPrevPose = poses(:, :, idxFrame - 2);
                    relMotion = inv(prevPrevPose) * prevPose;
                    relTransNorm = norm(relMotion(1:3, 4));
                    if relTransNorm > 0.01 && relTransNorm < 0.3
                        predictedPose = prevPose * relMotion;
                        rotDet = det(predictedPose(1:3, 1:3));
                        if abs(rotDet - 1.0) < 0.01
                            initialPoseForTracking = predictedPose;
                        end
                    end
                end
                
                warnState = warning('query', 'all');
                warning('off', 'all');
                trackArgs = structToNameValue(trackingOpts);
                trackRes = trackAgainstStaticMap(staticMap, initialPoseForTracking, pointsCam, segRes.staticMask, trackArgs{:});
                warning(warnState);
            end
            
            if ~trackRes.success
                fprintf('  Tracking failed. Static map size: %d points\n', size(staticMap.positions, 2));
                if isfield(trackRes, 'inlierMask')
                    fprintf('  Inliers: %d (required: %d)\n', nnz(trackRes.inlierMask), trackingOpts.MinInliers);
                end
                if isfield(trackRes, 'rmse')
                    fprintf('  RMSE: %.4f\n', trackRes.rmse);
                end
            end

            if trackRes.success
                currentPose = trackRes.Twc;
                fprintf('  Pose tracking succeeded (RMSE: %.4f, inliers: %d)\n', ...
                    trackRes.rmse, nnz(trackRes.inlierMask));
                
                if idxFrame > 2
                    prevPose = poses(:, :, idxFrame - 1);
                    relMotion = inv(prevPose) * currentPose;
                    relTrans = relMotion(1:3, 4);
                    relTransNorm = norm(relTrans);
                    
                    if relTransNorm > 0.5
                        warning('Frame %d: Large relative translation (%.4f m)', frameID, relTransNorm);
                    end
                    
                    rotDet = det(currentPose(1:3, 1:3));
                    if abs(rotDet - 1.0) > 0.01
                        warning('Frame %d: Tracked pose rotation matrix determinant is %.6f', frameID, rotDet);
                    end
                end
                
                if ENABLE_POSE_DIAGNOSTICS && ENABLE_EVALUATION
                    [~,gtPoseIdx] = min(abs(data.rgblist.time_posix(frameID) - data.poseTruth.timestamp));
                    poseRow_gt = data.poseTruth(gtPoseIdx, :);
                    txyz_gt = poseRow_gt.txyz;
                    quat_gt = poseRow_gt.quat;
                    quat_tum_gt = [quat_gt(2), quat_gt(3), quat_gt(4), quat_gt(1)];
                    gtPose = tumPoseToTransform(txyz_gt, quat_tum_gt);
                    poseDiff = norm(currentPose(1:3, 4) - gtPose(1:3, 4));
                    fprintf('  [Diagnostic] Pose diff from GT: translation=%.4f m\n', poseDiff);
                    
                    if trackRes.rmse > 0.1 || nnz(trackRes.inlierMask) < 100
                        warning('Tracking quality may be poor: RMSE=%.4f, inliers=%d', ...
                            trackRes.rmse, nnz(trackRes.inlierMask));
                    end
                    if poseDiff > 0.15
                        warning('Tracking result deviates significantly from GT (%.4f m)', poseDiff);
                    end
                end
            else
                warning('Pose tracking failed for frame %d.', frameID);
                
                if idxFrame > 2
                    prevPose = poses(:, :, idxFrame - 1);
                    prevPrevPose = poses(:, :, idxFrame - 2);
                    
                    relMotion = inv(prevPrevPose) * prevPose;
                    relTransNorm = norm(relMotion(1:3, 4));
                    
                    if relTransNorm > 0.01 && relTransNorm < 0.3
                        predictedPose = prevPose * relMotion;
                        rotDet = det(predictedPose(1:3, 1:3));
                        if abs(rotDet - 1.0) < 0.01
                            currentPose = predictedPose;
                            fprintf('  Using motion prediction (tracking failed, motion: %.4f m).\n', relTransNorm);
                        else
                            currentPose = prevPose;
                            fprintf('  Keeping previous pose (motion prediction invalid).\n');
                        end
                    else
                        currentPose = prevPose;
                        fprintf('  Keeping previous pose (previous motion unreasonable: %.4f m).\n', relTransNorm);
                    end
                else
                    currentPose = poses(:, :, idxFrame - 1);
                    fprintf('  Keeping previous pose (no motion model available).\n');
                end
                
                if ENABLE_POSE_DIAGNOSTICS && ENABLE_EVALUATION
                    [~,gtPoseIdx] = min(abs(data.rgblist.time_posix(frameID) - data.poseTruth.timestamp));
                    poseRow_gt = data.poseTruth(gtPoseIdx, :);
                    txyz_gt = poseRow_gt.txyz;
                    quat_gt = poseRow_gt.quat;
                    quat_tum_gt = [quat_gt(2), quat_gt(3), quat_gt(4), quat_gt(1)];
                    gtPose = tumPoseToTransform(txyz_gt, quat_tum_gt);
                    poseDiff = norm(currentPose(1:3, 4) - gtPose(1:3, 4));
                    fprintf('  [Diagnostic] Pose diff from GT: translation=%.4f m\n', poseDiff);
                    if poseDiff > 0.5
                        warning('Pose error is large (%.2f m)', poseDiff);
                    end
                end
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
        
        if ENABLE_POSE_DIAGNOSTICS && idxFrame > 2
            prevPose = poses(:, :, idxFrame - 1);
            relMotion = inv(prevPose) * currentPose;
            relTrans = relMotion(1:3, 4);
            relTransNorm = norm(relTrans);
            
            if relTransNorm > 0.5
                warning('Frame %d: Large relative translation (%.4f m)', frameID, relTransNorm);
            end
            
            rotDet = det(Rw);
            if abs(rotDet - 1.0) > 0.01
                warning('Frame %d: Rotation matrix determinant is %.6f', frameID, rotDet);
            end
        end
        
        lastStaticWorld = Rw * staticPoints + tw;
        lastDynamicWorld = Rw * dynamicPoints + tw;
        lastFrameID = frameID;

        fprintf('  Updating static map (%d static points, %d dynamic points)...\n', ...
            size(staticPoints, 2), size(dynamicPoints, 2));
        
        if ENABLE_POSE_DIAGNOSTICS && ~isempty(staticPoints)
            if any(isnan(lastStaticWorld(:))) || any(isinf(lastStaticWorld(:)))
                error('Frame %d: NaN or Inf in transformed static points!', frameID);
            end
        end
        
        totalPoints = size(staticPoints, 2) + size(dynamicPoints, 2);
        staticRatio = size(staticPoints, 2) / max(totalPoints, 1);
        
        if staticRatio > 0.85 && totalPoints > 1000
            warning('Frame %d: Suspiciously high static ratio (%.2f%%)', frameID, staticRatio * 100);
        end
        
        if ~isempty(staticPoints)
            staticMap = updateStaticMap(staticMap, currentPose, staticPoints, staticNormalsCam, staticColours);
            
            if ENABLE_POSE_DIAGNOSTICS
                if any(isnan(staticMap.positions(:))) || any(isinf(staticMap.positions(:)))
                    error('Frame %d: Static map contains NaN or Inf after update!', frameID);
                end
                fprintf('  Map size after update: %d surfels\n', size(staticMap.positions, 2));
            end
            
            if mod(idxFrame, 10) == 0 && ~isempty(staticMap.confidence)
                minConfidence = 0.5;
                keepMask = staticMap.confidence >= minConfidence;
                if nnz(~keepMask) > 0
                    fprintf('  Cleaning map: removing %d low-confidence surfels (confidence < %.2f)\n', ...
                        nnz(~keepMask), minConfidence);
                    staticMap.positions = staticMap.positions(:, keepMask);
                    staticMap.normals = staticMap.normals(:, keepMask);
                    staticMap.colours = staticMap.colours(:, keepMask);
                    staticMap.confidence = staticMap.confidence(keepMask);
                    staticMap.radius = staticMap.radius(keepMask);
                end
            end
        else
            warning('Frame %d: No static points to update map.', frameID);
        end

        dynamicCounts(idxFrame) = nnz(segRes.dynamicMask);
        staticCounts(idxFrame) = nnz(segRes.staticMask);
        poses(:, :, idxFrame) = currentPose;
        trajectory(:, idxFrame) = currentPose(1:3, 4);
        
        if USE_FRAME_TO_FRAME
            Rw = currentPose(1:3, 1:3);
            tw = currentPose(1:3, 4);
            
            staticPointsCam = pointsCam(:, segRes.staticMask);
            staticNormalsCam = normalsCam(:, segRes.staticMask);
            if size(staticPointsCam, 2) > 50
                prevPointsWorld = Rw * staticPointsCam + tw;
                prevNormalsWorld = normalizeColumns(Rw * staticNormalsCam);
            else
                if isempty(prevPointsWorld) || size(prevPointsWorld, 2) < 100
                    prevPointsWorld = Rw * pointsCam + tw;
                    prevNormalsWorld = normalizeColumns(Rw * normalsCam);
                end
            end
        end
        
        if ENABLE_EVALUATION
            [~,gtPoseIdx] = min(abs(data.rgblist.time_posix(frameID) - data.poseTruth.timestamp));
            poseRow_gt = data.poseTruth(gtPoseIdx, :);
            txyz_gt = poseRow_gt.txyz;
            quat_gt = poseRow_gt.quat;
            quat_tum_gt = [quat_gt(2), quat_gt(3), quat_gt(4), quat_gt(1)];
            gtPoses(:, :, idxFrame) = tumPoseToTransform(txyz_gt, quat_tum_gt);
            gtTrajectory(:, idxFrame) = gtPoses(1:3, 4, idxFrame);
        end

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
function frameData = loadFrameDataBonn(data, frameRange, localIdx, depthFilterOpts, denoiseOpts, normalOpts)
    frameID = frameRange(localIdx);

    rgb_path = data.image{localIdx};
    depth_path = data.depth{localIdx};
    
    if ~exist(rgb_path, 'file')
        warning('RGB file not found: %s', rgb_path);
        frameData = struct('frameID', frameID, 'XYZcam', zeros(3,0), 'RGB', zeros(3,0), 'normalsCam', zeros(3,0));
        return;
    end
    
    if ~exist(depth_path, 'file')
        warning('Depth file not found: %s', depth_path);
        frameData = struct('frameID', frameID, 'XYZcam', zeros(3,0), 'RGB', zeros(3,0), 'normalsCam', zeros(3,0));
        return;
    end
    
    image = imread(rgb_path);
    depth = depthReadTUM(depth_path);

    [XYZcam, RGB] = depthToPointCloudBonn(image, depth, 'depthFilter', depthFilterOpts);
    
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
    numFrames = size(poses, 3);
    ate_translation = zeros(1, numFrames);
    ate_rotation = zeros(1, numFrames);
    
    for i = 1:numFrames
        t_est = poses(1:3, 4, i);
        t_gt = gtPoses(1:3, 4, i);
        ate_translation(i) = norm(t_est - t_gt);
        
        R_est = poses(1:3, 1:3, i);
        R_gt = gtPoses(1:3, 1:3, i);
        R_diff = R_est * R_gt';
        trace_R = trace(R_diff);
        trace_R = max(-1, min(1, trace_R));
        angle_rad = acos((trace_R - 1) / 2);
        ate_rotation(i) = rad2deg(angle_rad);
    end
    
    fprintf('  Average Translation Error: %.4f m\n', mean(ate_translation));
    fprintf('  Average Rotation Error: %.4f deg\n', mean(ate_rotation));
    fprintf('  Max Translation Error: %.4f m\n', max(ate_translation));
    fprintf('  Max Rotation Error: %.4f deg\n', max(ate_rotation));
    fprintf('  RMSE Translation Error: %.4f m\n', sqrt(mean(ate_translation.^2)));
    
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
    plot(frameRange, ate_rotation / 10, 'r-', 'LineWidth', 2);
    xlabel('Frame ID');
    ylabel('Error');
    title('Error Over Time');
    legend('Translation (m)', 'Rotation (deg/10)', 'Location', 'best');
    grid on;
end
