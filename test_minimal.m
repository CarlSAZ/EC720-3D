%% MINIMAL TEST
clear; close all; clc;

fprintf('=== MINIMAL TEST START ===\n');

scriptDir = fileparts(mfilename('fullpath'));
if exist('initializeStaticMap', 'file') ~= 2
    startupPath = fullfile(scriptDir, 'startup.m');
    if exist(startupPath, 'file')
        run(startupPath);
    end
end

fprintf('Step 1: Check dependencies...\n');
if ~exist('knnsearch', 'builtin') && ~exist('knnsearch', 'file')
    error('knnsearch not found');
end
fprintf('  OK\n');

fprintf('Step 2: Load SUN3D data (frame 1 only)...\n');
try
    sequenceName = 'hotel_umd/maryland_hotel3';
    data = loadSUN3D(sequenceName, 1);
    fprintf('  OK - Loaded frame 1\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    return;
end

fprintf('Step 3: Convert to point cloud...\n');
try
    image = imread(data.image{1});
    depth = double(depthRead(data.depth{1}));
    depthFilterOpts = struct('minDepth', 0.4, 'maxDepth', 4.5, 'medianKernel', 3);
    [XYZcam, RGB] = depthToPointCloud(image, depth, data.K, 'depthFilter', depthFilterOpts);
    fprintf('  OK - Points: %d\n', size(XYZcam, 2));
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    return;
end

fprintf('Step 4: Estimate normals...\n');
try
    normalOpts = struct('K', 25);
    normalsCam = estimateNormals(XYZcam, 'K', normalOpts.K, 'ViewPoint', [0; 0; 0]);
    fprintf('  OK - Normals: %d\n', size(normalsCam, 2));
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    return;
end

fprintf('Step 5: Initialize static map...\n');
try
    firstPose = data.extrinsicsC2W(:, :, 1);
    if size(firstPose, 1) == 3
        firstPose = [firstPose; 0 0 0 1];
    end
    Rw = firstPose(1:3, 1:3);
    normalsWorld = normalizeColumns(Rw * normalsCam);
    XYZworld = Rw * XYZcam + firstPose(1:3, 4);
    pcInit = struct('XYZ', XYZworld, 'RGB', RGB);
    staticMap = initializeStaticMap(pcInit, normalsWorld);
    fprintf('  OK - Map surfels: %d\n', size(staticMap.positions, 2));
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    return;
end

fprintf('Step 6: Predict static map...\n');
try
    prediction = predictStaticMap(staticMap, firstPose);
    fprintf('  OK - Predicted points: %d\n', size(prediction.XYZcam, 2));
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    return;
end

fprintf('Step 7: Estimate static mask (with small subset)...\n');
try
    sampleIdx = 1:10:size(XYZcam, 2);
    if isempty(sampleIdx)
        error('No points to sample');
    end
    pointsCam = XYZcam(:, sampleIdx);
    colours = RGB(:, sampleIdx);
    segRes = estimateStaticMask(pointsCam, colours, prediction);
    fprintf('  OK - Static: %d, Dynamic: %d\n', nnz(segRes.staticMask), nnz(segRes.dynamicMask));
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    fprintf('  Stack: %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    return;
end

fprintf('Step 8: Track against static map...\n');
try
    trackRes = trackAgainstStaticMap(staticMap, firstPose, pointsCam, segRes.staticMask, ...
        'MaxIterations', 5, 'MinInliers', 10);
    fprintf('  OK - Success: %d, RMSE: %.4f\n', trackRes.success, trackRes.rmse);
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    fprintf('  Stack: %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    return;
end

fprintf('Step 9: Update static map...\n');
try
    staticPoints = pointsCam(:, segRes.staticMask);
    staticNormalsCam = normalsCam(:, sampleIdx(segRes.staticMask));
    staticColours = colours(:, segRes.staticMask);
    if ~isempty(staticPoints)
        staticMap = updateStaticMap(staticMap, firstPose, staticPoints, staticNormalsCam, staticColours);
        fprintf('  OK - Updated map surfels: %d\n', size(staticMap.positions, 2));
    else
        fprintf('  SKIPPED - No static points\n');
    end
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    fprintf('  Stack: %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    return;
end

fprintf('\n=== ALL TESTS PASSED ===\n');

function normals = normalizeColumns(vectors)
norms = sqrt(sum(vectors.^2, 1));
norms(norms < eps) = 1;
normals = vectors ./ norms;
end

