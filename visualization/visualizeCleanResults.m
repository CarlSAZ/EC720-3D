function visualizeCleanResults(staticMap, trajectory, varargin)
%VISUALIZECLEANRESULTS Create a clean visualization without black lines and dark points.
%   visualizeCleanResults(staticMap, trajectory) creates a clean 3D visualization
%   of the static map and trajectory, filtering out:
%   - Large trajectory jumps (disconnected segments)
%   - Dark/black points in the point cloud
%
%   Name-value pairs:
%       'DownsampleMap'      - downsample factor for map (default 5)
%       'MaxTrajectoryJump'   - maximum allowed jump distance in meters (default 2.0)
%       'MinPointBrightness'  - minimum brightness threshold [0,1] (default 0.15)
%       'FigureNumber'        - figure number to use (default 200)
%       'Title'               - figure title (default 'Clean Static Map with Trajectory')

narginchk(2, inf);

parser = inputParser;
addParameter(parser, 'DownsampleMap', 5, @(x) isnumeric(x) && x > 0);
addParameter(parser, 'MaxTrajectoryJump', 2.0, @(x) isnumeric(x) && x > 0);
addParameter(parser, 'MinPointBrightness', 0.15, @(x) isnumeric(x) && x >= 0 && x <= 1);
addParameter(parser, 'FigureNumber', 200, @(x) isnumeric(x) && x > 0);
addParameter(parser, 'Title', 'Clean Static Map with Trajectory', @ischar);
parse(parser, varargin{:});
opts = parser.Results;

% Validate inputs
if isempty(staticMap) || ~isfield(staticMap, 'positions') || isempty(staticMap.positions)
    warning('visualizeCleanResults:EmptyMap', 'Static map is empty.');
    return;
end

if isempty(trajectory) || size(trajectory, 2) < 1
    warning('visualizeCleanResults:InvalidTrajectory', 'Trajectory is invalid.');
    return;
end

% Create figure
fig = figure(opts.FigureNumber);
clf(fig);
set(fig, 'Name', opts.Title, 'Color', 'white', 'Position', [100, 100, 1200, 900]);
ax = axes('Parent', fig);
hold(ax, 'on');

% Filter and plot static map
fprintf('[visualizeCleanResults] Processing static map (%d points)...\n', size(staticMap.positions, 2));

% Step 1: Filter dark points
mapIdx = 1:opts.DownsampleMap:size(staticMap.positions, 2);
mapIdx = mapIdx(mapIdx <= size(staticMap.positions, 2));

    if isfield(staticMap, 'colours') && ~isempty(staticMap.colours) && ...
   size(staticMap.colours, 2) >= max(mapIdx)
    % Calculate brightness for each point
    brightness = mean(staticMap.colours(:, mapIdx), 1);
    % Filter out dark points
    brightMask = brightness >= opts.MinPointBrightness;
    filteredMapIdx = mapIdx(brightMask);
    
    fprintf('[visualizeCleanResults] Filtered %d dark points, keeping %d bright points.\n', ...
        numel(mapIdx) - numel(filteredMapIdx), numel(filteredMapIdx));
else
    filteredMapIdx = mapIdx;
    fprintf('[visualizeCleanResults] No color information, using all points.\n');
end

% Plot filtered point cloud
if ~isempty(filteredMapIdx)
    if isfield(staticMap, 'colours') && ~isempty(staticMap.colours) && ...
       size(staticMap.colours, 2) >= max(filteredMapIdx)
        scatter3(ax, staticMap.positions(1, filteredMapIdx), ...
            staticMap.positions(2, filteredMapIdx), ...
            staticMap.positions(3, filteredMapIdx), ...
            8, staticMap.colours(:, filteredMapIdx)', '.');
    else
        scatter3(ax, staticMap.positions(1, filteredMapIdx), ...
            staticMap.positions(2, filteredMapIdx), ...
            staticMap.positions(3, filteredMapIdx), ...
            8, [0.5, 0.5, 0.8], '.');
    end
    fprintf('[visualizeCleanResults] Plotted %d map points.\n', numel(filteredMapIdx));
else
    warning('visualizeCleanResults:NoPoints', 'No points to plot after filtering.');
end

% Filter and plot trajectory
fprintf('[visualizeCleanResults] Processing trajectory (%d points)...\n', size(trajectory, 2));

if size(trajectory, 2) >= 2
    % Calculate distances between consecutive trajectory points
    trajDiff = diff(trajectory, 1, 2);
    distances = sqrt(sum(trajDiff.^2, 1));
    
    % Find valid segments (where jump is less than threshold)
    validMask = distances < opts.MaxTrajectoryJump;
    
    % Plot trajectory in segments
    if any(validMask)
        % Find continuous segments
        segments = findContinuousSegments(validMask);
        
        fprintf('[visualizeCleanResults] Found %d trajectory segments.\n', size(segments, 1));
        
        % Plot each valid segment
        for i = 1:size(segments, 1)
            startIdx = segments(i, 1);
            endIdx = segments(i, 2) + 1;  % +1 because diff reduces size by 1
            if endIdx > size(trajectory, 2)
                endIdx = size(trajectory, 2);
            end
            
            plot3(ax, trajectory(1, startIdx:endIdx), ...
                trajectory(2, startIdx:endIdx), ...
                trajectory(3, startIdx:endIdx), ...
                'b-', 'LineWidth', 2.5, 'DisplayName', 'Camera Trajectory');
        end
        
        % Plot start and end points
        scatter3(ax, trajectory(1, 1), trajectory(2, 1), trajectory(3, 1), ...
            150, 'go', 'filled', 'LineWidth', 2, 'MarkerEdgeColor', 'k', ...
            'DisplayName', 'Start');
        scatter3(ax, trajectory(1, end), trajectory(2, end), trajectory(3, end), ...
            150, 'ro', 'filled', 'LineWidth', 2, 'MarkerEdgeColor', 'k', ...
            'DisplayName', 'End');
    else
        % If no valid segments, just plot points
        fprintf('[visualizeCleanResults] No valid trajectory segments, plotting points only.\n');
        scatter3(ax, trajectory(1, :), trajectory(2, :), trajectory(3, :), ...
            50, 'b', 'filled', 'DisplayName', 'Trajectory Points');
    end
elseif size(trajectory, 2) == 1
    % Single point
    scatter3(ax, trajectory(1, 1), trajectory(2, 1), trajectory(3, 1), ...
        150, 'go', 'filled', 'LineWidth', 2, 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'Start');
end

% Set axes properties
title(ax, opts.Title, 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax, 'X (m)', 'FontSize', 12);
ylabel(ax, 'Y (m)', 'FontSize', 12);
zlabel(ax, 'Z (m)', 'FontSize', 12);
axis(ax, 'equal');
grid(ax, 'on');
view(ax, 135, 30);
legend(ax, 'Location', 'best', 'FontSize', 10);

% Add statistics text
statsText = {
    sprintf('Map Points: %d (displayed: %d)', size(staticMap.positions, 2), numel(filteredMapIdx));
    sprintf('Trajectory Points: %d', size(trajectory, 2));
    if size(trajectory, 2) >= 2
        totalLength = sum(distances(validMask));
        fprintf('[visualizeCleanResults] Total trajectory length: %.2f m\n', totalLength);
        statsText{end+1} = sprintf('Trajectory Length: %.2f m', totalLength);
    end
};

% Add text box with statistics
annotation(fig, 'textbox', [0.02, 0.02, 0.25, 0.15], ...
    'String', statsText, ...
    'FontSize', 10, ...
    'BackgroundColor', 'white', ...
    'EdgeColor', 'black', ...
    'LineWidth', 1);

set(fig, 'Visible', 'on');
drawnow;

fprintf('[visualizeCleanResults] Clean visualization complete.\n');

end

%% Helper function to find continuous segments
function segments = findContinuousSegments(mask)
% FindContinuousSegments Find continuous true segments in a logical mask.
%   segments = findContinuousSegments(mask) returns an Nx2 matrix where
%   each row [start, end] represents a continuous segment of true values.

if isempty(mask) || ~any(mask)
    segments = zeros(0, 2);
    return;
end

% Find transitions
diffMask = diff([false, mask, false]);
starts = find(diffMask == 1);
ends = find(diffMask == -1) - 1;

segments = [starts(:), ends(:)];

end

