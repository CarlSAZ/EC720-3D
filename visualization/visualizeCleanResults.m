function visualizeCleanResults(staticMap, trajectory, varargin)
% Create clean visualization filtering out dark points and trajectory jumps

narginchk(2, inf);

parser = inputParser;
addParameter(parser, 'DownsampleMap', 5, @(x) isnumeric(x) && x > 0);
addParameter(parser, 'MaxTrajectoryJump', 2.0, @(x) isnumeric(x) && x > 0);
addParameter(parser, 'MinPointBrightness', 0.15, @(x) isnumeric(x) && x >= 0 && x <= 1);
addParameter(parser, 'FigureNumber', 200, @(x) isnumeric(x) && x > 0);
addParameter(parser, 'Title', 'Clean Static Map with Trajectory', @ischar);
parse(parser, varargin{:});
opts = parser.Results;

if isempty(staticMap) || ~isfield(staticMap, 'positions') || isempty(staticMap.positions)
    warning('visualizeCleanResults:EmptyMap', 'Static map is empty.');
    return;
end

if isempty(trajectory) || size(trajectory, 2) < 1
    warning('visualizeCleanResults:InvalidTrajectory', 'Trajectory is invalid.');
    return;
end

fig = figure(opts.FigureNumber);
clf(fig);
set(fig, 'Name', opts.Title, 'Color', 'white', 'Position', [100, 100, 1200, 900]);
ax = axes('Parent', fig);
hold(ax, 'on');

fprintf('[visualizeCleanResults] Processing static map (%d points)...\n', size(staticMap.positions, 2));

mapIdx = 1:opts.DownsampleMap:size(staticMap.positions, 2);
mapIdx = mapIdx(mapIdx <= size(staticMap.positions, 2));

    if isfield(staticMap, 'colours') && ~isempty(staticMap.colours) && ...
   size(staticMap.colours, 2) >= max(mapIdx)
    brightness = mean(staticMap.colours(:, mapIdx), 1);
    brightMask = brightness >= opts.MinPointBrightness;
    filteredMapIdx = mapIdx(brightMask);
    
    fprintf('[visualizeCleanResults] Filtered %d dark points, keeping %d bright points.\n', ...
        numel(mapIdx) - numel(filteredMapIdx), numel(filteredMapIdx));
else
    filteredMapIdx = mapIdx;
    fprintf('[visualizeCleanResults] No color information, using all points.\n');
end

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

fprintf('[visualizeCleanResults] Processing trajectory (%d points)...\n', size(trajectory, 2));

if size(trajectory, 2) >= 2
    trajDiff = diff(trajectory, 1, 2);
    distances = sqrt(sum(trajDiff.^2, 1));
    
    validMask = distances < opts.MaxTrajectoryJump;
    
    if any(validMask)
        segments = findContinuousSegments(validMask);
        
        fprintf('[visualizeCleanResults] Found %d trajectory segments.\n', size(segments, 1));
        
        for i = 1:size(segments, 1)
            startIdx = segments(i, 1);
            endIdx = segments(i, 2) + 1;
            if endIdx > size(trajectory, 2)
                endIdx = size(trajectory, 2);
            end
            
            plot3(ax, trajectory(1, startIdx:endIdx), ...
                trajectory(2, startIdx:endIdx), ...
                trajectory(3, startIdx:endIdx), ...
                'b-', 'LineWidth', 2.5,                 'DisplayName', 'Camera Trajectory');
        end
        
        scatter3(ax, trajectory(1, 1), trajectory(2, 1), trajectory(3, 1), ...
            150, 'go', 'filled', 'LineWidth', 2, 'MarkerEdgeColor', 'k', ...
            'DisplayName', 'Start');
        scatter3(ax, trajectory(1, end), trajectory(2, end), trajectory(3, end), ...
            150, 'ro', 'filled', 'LineWidth', 2, 'MarkerEdgeColor', 'k', ...
            'DisplayName', 'End');
    else
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

title(ax, opts.Title, 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax, 'X (m)', 'FontSize', 12);
ylabel(ax, 'Y (m)', 'FontSize', 12);
zlabel(ax, 'Z (m)', 'FontSize', 12);
axis(ax, 'equal');
grid(ax, 'on');
view(ax, 135, 30);
legend(ax, 'Location', 'best', 'FontSize', 10);

statsText = {
    sprintf('Map Points: %d (displayed: %d)', size(staticMap.positions, 2), numel(filteredMapIdx));
    sprintf('Trajectory Points: %d', size(trajectory, 2));
    if size(trajectory, 2) >= 2
        totalLength = sum(distances(validMask));
        fprintf('[visualizeCleanResults] Total trajectory length: %.2f m\n', totalLength);
        statsText{end+1} = sprintf('Trajectory Length: %.2f m', totalLength);
    end
};

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

function segments = findContinuousSegments(mask)

if isempty(mask) || ~any(mask)
    segments = zeros(0, 2);
    return;
end

diffMask = diff([false, mask, false]);
starts = find(diffMask == 1);
ends = find(diffMask == -1) - 1;

segments = [starts(:), ends(:)];

end

