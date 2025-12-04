function visualizeProgress(staticMap, currentPose, pointsCam, staticMask, dynamicMask, trajectory, varargin)
%VISUALIZEPROGRESS Show real-time progress during StaticFusion processing.
%   visualizeProgress(staticMap, currentPose, pointsCam, staticMask, dynamicMask, trajectory)
%   displays the current state of the static map and frame segmentation.
%
%   Optional name-value pairs:
%       'DownsampleMap'    - downsample factor for map (default 3)
%       'DownsampleFrame'  - downsample factor for frame points (default 5)
%       'FigureNumber'     - figure number to use (default 100)

narginchk(6, inf);

% Parse options
p = inputParser;
addParameter(p, 'DownsampleMap', 3, @(x) isnumeric(x) && x > 0);
addParameter(p, 'DownsampleFrame', 5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'FigureNumber', 100, @(x) isnumeric(x) && x > 0);
parse(p, varargin{:});
opts = p.Results;

% Validate inputs
if isempty(staticMap) || ~isfield(staticMap, 'positions') || isempty(staticMap.positions)
    return;
end
if isempty(pointsCam) || size(pointsCam, 2) == 0
    return;
end

try
    % Transform points to world coordinates
    Rw = currentPose(1:3, 1:3);
    tw = currentPose(1:3, 4);

    staticWorld = [];
    dynamicWorld = [];
    if nnz(staticMask) > 0 && numel(staticMask) == size(pointsCam, 2)
        staticWorld = Rw * pointsCam(:, staticMask) + tw;
    end
    if nnz(dynamicMask) > 0 && numel(dynamicMask) == size(pointsCam, 2)
        dynamicWorld = Rw * pointsCam(:, dynamicMask) + tw;
    end
catch ME
    warning('visualizeProgress:TransformFailed', 'Failed to transform points: %s', ME.message);
    return;
end

% Create or clear figure
fig = figure(opts.FigureNumber);
clf(fig);
set(fig, 'Visible', 'on', 'Name', 'StaticFusion Progress');
drawnow;

% Left subplot: Static map with trajectory
subplot(1, 2, 1);
hold on;
mapIdx = 1:opts.DownsampleMap:size(staticMap.positions, 2);
if ~isempty(mapIdx) && max(mapIdx) <= size(staticMap.positions, 2)
    try
        if isfield(staticMap, 'colours') && size(staticMap.colours, 2) >= max(mapIdx)
            scatter3(staticMap.positions(1, mapIdx), ...
                staticMap.positions(2, mapIdx), ...
                staticMap.positions(3, mapIdx), ...
                4, staticMap.colours(:, mapIdx)', '.');
        else
            scatter3(staticMap.positions(1, mapIdx), ...
                staticMap.positions(2, mapIdx), ...
                staticMap.positions(3, mapIdx), ...
                4, 'b', '.');
        end
    catch
        scatter3(staticMap.positions(1, mapIdx), ...
            staticMap.positions(2, mapIdx), ...
            staticMap.positions(3, mapIdx), ...
            4, 'b', '.');
    end
end

if size(trajectory, 2) >= 2
    plot3(trajectory(1, :), trajectory(2, :), trajectory(3, :), 'k-', 'LineWidth', 1.5);
end
title('Static Map (downsampled) with Trajectory');
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
view(135, 30);

% Right subplot: Current frame segmentation
subplot(1, 2, 2);
hold on;
if ~isempty(staticWorld) && size(staticWorld, 2) > 0
    idx = 1:opts.DownsampleFrame:size(staticWorld, 2);
    scatter3(staticWorld(1, idx), staticWorld(2, idx), staticWorld(3, idx), ...
        12, 'g', 'filled');
end
if ~isempty(dynamicWorld) && size(dynamicWorld, 2) > 0
    idx = 1:opts.DownsampleFrame:size(dynamicWorld, 2);
    scatter3(dynamicWorld(1, idx), dynamicWorld(2, idx), dynamicWorld(3, idx), ...
        12, 'r', 'filled');
end
if size(trajectory, 2) >= 2
    plot3(trajectory(1, :), trajectory(2, :), trajectory(3, :), 'k-', 'LineWidth', 1.5);
end
title(sprintf('Current Frame: static (green, %d) vs dynamic (red, %d)', ...
    nnz(staticMask), nnz(dynamicMask)));
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
view(135, 30);

set(fig, 'Visible', 'on');
drawnow;
pause(0.05);
end

