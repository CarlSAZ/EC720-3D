function data = loadBonn(sequencePath, frameIDs)
%LOADBONN Load Bonn RGB-D dataset sequence.
%   data = LOADBONN(sequencePath, frameIDs) loads a Bonn RGB-D sequence
%   from the specified path.
%
%   Inputs:
%       sequencePath - Path to the Bonn sequence directory (should contain
%                      rgb.txt, depth.txt, groundtruth.txt, rgb/, depth/)
%       frameIDs     - Optional. Frame indices to load (1-based). If empty
%                      or not provided, loads all frames.
%
%   Output:
%       data struct with fields:
%           basedir      - Base directory of the sequence
%           rgblist      - Table with columns: time_posix, filename
%           depthlist    - Table with columns: time_posix, filename
%           poseTruth    - Table with columns: timestamp, txyz, quat
%           K            - Camera intrinsic matrix (3x3)
%           image        - Cell array of RGB image paths (for requested frames)
%           depth        - Cell array of depth image paths (for requested frames)
%           extrinsicsC2W - 4x4xN array of camera-to-world transforms (optional)
%
%   Example:
%       data = loadBonn('/path/to/rgbd_bonn_crowd', 1:10);
%       data = loadBonn('/path/to/rgbd_bonn_crowd');  % Load all frames

narginchk(1, 2);

% Validate sequence path
if ~exist('sequencePath', 'var') || isempty(sequencePath)
    error('loadBonn:InvalidPath', 'Sequence path must be provided.');
end

if ~exist(sequencePath, 'dir')
    error('loadBonn:PathNotFound', 'Sequence directory not found: %s', sequencePath);
end

data.basedir = sequencePath;

% Camera intrinsics for Bonn dataset (hardcoded)
% fx = 542.822841, fy = 542.576870, cx = 315.593520, cy = 237.756098
data.K = [542.822841, 0,           315.593520;
          0,           542.576870, 237.756098;
          0,           0,           1];

% Load RGB file list
rgb_txt_path = fullfile(sequencePath, 'rgb.txt');
if ~exist(rgb_txt_path, 'file')
    error('loadBonn:RGBFileNotFound', 'rgb.txt not found in: %s', sequencePath);
end

fid = fopen(rgb_txt_path, 'r');
if fid == -1
    error('loadBonn:CannotOpenRGB', 'Cannot open rgb.txt: %s', rgb_txt_path);
end
temp = textscan(fid, '%f %s', 'HeaderLines', 2);
fclose(fid);

if isempty(temp{1}) || isempty(temp{2})
    error('loadBonn:EmptyRGB', 'rgb.txt is empty or invalid.');
end

data.rgblist = table(temp{1}, string(temp{2}), 'VariableNames', {'time_posix', 'filename'});
numRGBFrames = height(data.rgblist);

% Load depth file list
depth_txt_path = fullfile(sequencePath, 'depth.txt');
if ~exist(depth_txt_path, 'file')
    error('loadBonn:DepthFileNotFound', 'depth.txt not found in: %s', sequencePath);
end

fid = fopen(depth_txt_path, 'r');
if fid == -1
    error('loadBonn:CannotOpenDepth', 'Cannot open depth.txt: %s', depth_txt_path);
end
temp = textscan(fid, '%f %s', 'HeaderLines', 2);
fclose(fid);

if isempty(temp{1}) || isempty(temp{2})
    error('loadBonn:EmptyDepth', 'depth.txt is empty or invalid.');
end

data.depthlist = table(temp{1}, string(temp{2}), 'VariableNames', {'time_posix', 'filename'});
numDepthFrames = height(data.depthlist);

% Load ground truth poses
gt_txt_path = fullfile(sequencePath, 'groundtruth.txt');
if ~exist(gt_txt_path, 'file')
    warning('loadBonn:GTFileNotFound', 'groundtruth.txt not found. Setting poseTruth to empty.');
    data.poseTruth = table();
    data.extrinsicsC2W = [];
else
    fid = fopen(gt_txt_path, 'r');
    if fid == -1
        warning('loadBonn:CannotOpenGT', 'Cannot open groundtruth.txt. Setting poseTruth to empty.');
        data.poseTruth = table();
        data.extrinsicsC2W = [];
    else
        % Format: timestamp tx ty tz qx qy qz qw
        temp = textscan(fid, '%f %f %f %f %f %f %f %f', 'HeaderLines', 2);
        fclose(fid);
        
        if ~isempty(temp{1}) && ~isempty(temp{2})
            % Store as: timestamp, txyz (Nx3 matrix), quat (Nx4 matrix)
            % Format compatible with bronnTransform function (same as testBonn.m)
            % Each row: [tx, ty, tz] and [qw, qx, qy, qz]
            data.poseTruth = table(temp{1}, [temp{2:4}], [temp{8}, temp{5:7}], ...
                'VariableNames', {'timestamp', 'txyz', 'quat'});
            
            % Convert to extrinsicsC2W format (4x4xN) for compatibility
            numPoses = height(data.poseTruth);
            data.extrinsicsC2W = zeros(4, 4, numPoses);
            for i = 1:numPoses
                poseRow = data.poseTruth(i, :);
                try
                    T = bronnTransform(poseRow);
                    data.extrinsicsC2W(:, :, i) = T;
                catch ME
                    warning('loadBonn:TransformError', 'Failed to transform pose %d: %s', i, ME.message);
                    data.extrinsicsC2W(:, :, i) = eye(4);
                end
            end
        else
            data.poseTruth = table();
            data.extrinsicsC2W = [];
        end
    end
end

% Determine frame range
if ~exist('frameIDs', 'var') || isempty(frameIDs)
    frameIDs = 1:numRGBFrames;
end

% Validate frame IDs
if any(frameIDs < 1) || any(frameIDs > numRGBFrames)
    error('loadBonn:InvalidFrameIDs', ...
        'Frame IDs must be in range [1, %d]. Got range [%d, %d].', ...
        numRGBFrames, min(frameIDs), max(frameIDs));
end

% Build image and depth paths for requested frames
data.image = cell(1, numel(frameIDs));
data.depth = cell(1, numel(frameIDs));

for i = 1:numel(frameIDs)
    frameIdx = frameIDs(i);
    
    % RGB image path
    rgb_filename = data.rgblist.filename(frameIdx);
    data.image{i} = fullfile(sequencePath, rgb_filename);
    
    % Find corresponding depth frame by timestamp
    rgb_timestamp = data.rgblist.time_posix(frameIdx);
    [~, depthIdx] = min(abs(data.depthlist.time_posix - rgb_timestamp));
    depth_filename = data.depthlist.filename(depthIdx);
    data.depth{i} = fullfile(sequencePath, depth_filename);
end

% Print summary
fprintf('[loadBonn] Loaded Bonn sequence from: %s\n', sequencePath);
fprintf('  RGB frames: %d (requested: %d)\n', numRGBFrames, numel(frameIDs));
fprintf('  Depth frames: %d\n', numDepthFrames);
if ~isempty(data.poseTruth)
    fprintf('  Ground truth poses: %d\n', height(data.poseTruth));
else
    fprintf('  Ground truth poses: none\n');
end

end
