% Build and visualize static map structure from a single frame

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
frameIdx = 1;

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

%% Load data and build static map
fprintf('[VisualizeStaticMap] Loading SUN3D frame %d of %s...\n', frameIdx, sequenceName);
data = loadSUN3D(sequenceName, frameIdx);

if isempty(data.image) || isempty(data.depth)
    error('visualize_static_map:EmptyData', 'Failed to load image/depth for the requested frame.');
end

colorImg = imread(data.image{1});
depthImg = double(depthRead(data.depth{1}));

fprintf('[VisualizeStaticMap] Converting depth to point cloud...\n');
[XYZcam, RGB] = depthToPointCloud(colorImg, depthImg, data.K, 'depthFilter', depthFilterOpts);

if ~isempty(denoiseOpts)
    denoiseArgs = structToArgs(denoiseOpts);
    [XYZcam, RGB] = denoisePointCloud(XYZcam, RGB, denoiseArgs{:});
end

if isempty(XYZcam)
    error('visualize_static_map:EmptyPointCloud', 'Point cloud is empty after preprocessing.');
end

fprintf('[VisualizeStaticMap] Estimating normals...\n');
normalArgs = structToArgs(normalOpts);
normalsCam = estimateNormals(XYZcam, normalArgs{:}, 'ViewPoint', [0; 0; 0]);

if size(normalsCam, 2) ~= size(XYZcam, 2)
    warning('visualize_static_map:NormalMismatch', 'Normal count mismatch, using default normals.');
    normalsCam = zeros(3, size(XYZcam, 2));
    normalsCam(3, :) = 1;
end

extrinsic = ensureHomogeneousTransform(data.extrinsicsC2W(:, :, frameIdx));
Rwc = extrinsic(1:3, 1:3);
twc = extrinsic(1:3, 4);

XYZworld = Rwc * XYZcam + twc;
normalsWorld = normalizeColumns(Rwc * normalsCam);

fprintf('[VisualizeStaticMap] Initialising static map with %d surfels...\n', size(XYZworld, 2));
pcInit = struct('XYZ', XYZworld, 'RGB', RGB);
mapInitArgs = structToArgs(mapInitOpts);
map = initializeStaticMap(pcInit, normalsWorld, mapInitArgs{:});

%% Visualise the map structure
fprintf('[VisualizeStaticMap] Visualising static map structure...\n');
plotStaticMapStructure(map, 'Downsample', 5, 'Title', sprintf('Static Map (Frame %d)', frameIdx));

fprintf('[VisualizeStaticMap] Done.\n');
function args = structToArgs(s)
if isempty(s)
    args = {};
    return;
end
fn = fieldnames(s);
args = cell(1, numel(fn) * 2);
for i = 1:numel(fn)
    args{2*i - 1} = fn{i};
    args{2*i} = s.(fn{i});
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
