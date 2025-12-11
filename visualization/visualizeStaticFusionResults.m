function visualizeStaticFusionResults(staticMap, poses, trajectory, staticCounts, dynamicCounts, frameRange, varargin)
% Comprehensive visualization of StaticFusion results

narginchk(6, inf);

p = inputParser;
addParameter(p, 'ShowProgress', true, @islogical);
addParameter(p, 'ShowFinal', true, @islogical);
addParameter(p, 'ShowStatistics', true, @islogical);
addParameter(p, 'DownsampleMap', 5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'DownsampleFrame', 5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LastStaticWorld', [], @(x) isempty(x) || (isfloat(x) && size(x,1) == 3));
addParameter(p, 'LastDynamicWorld', [], @(x) isempty(x) || (isfloat(x) && size(x,1) == 3));
addParameter(p, 'LastFrameID', [], @(x) isempty(x) || isnumeric(x));
parse(p, varargin{:});
opts = p.Results;

if isempty(staticMap) || ~isfield(staticMap, 'positions') || isempty(staticMap.positions)
    warning('visualizeStaticFusionResults:EmptyMap', 'Static map is empty, skipping visualization.');
    return;
end

if isempty(trajectory) || size(trajectory, 2) < 2
    warning('visualizeStaticFusionResults:InvalidTrajectory', 'Trajectory is invalid, skipping visualization.');
    return;
end

if opts.ShowStatistics
    try
        figStats = figure('Name', 'Static vs Dynamic Counts', 'Visible', 'on');
        h1 = plot(frameRange, staticCounts, '-g', 'LineWidth', 1.5, 'DisplayName', 'Static (processed)'); hold on;
        h2 = plot(frameRange, dynamicCounts, '-r', 'LineWidth', 1.5, 'DisplayName', 'Dynamic (processed)');
        xlabel('Frame'); ylabel('Point count (sampled)');
        legend([h1, h2], {'Static (processed)', 'Dynamic (processed)'}, 'Location', 'best');
        grid on;
        title('Static vs Dynamic Counts Over Time');
        drawnow;
    catch ME
        warning('Statistics plot failed: %s', ME.message);
    end
end

if opts.ShowFinal
    if ~isempty(staticMap.positions) && size(staticMap.positions, 2) > 0 && ...
       ~isempty(trajectory) && size(trajectory, 2) >= 2
        try
            visualizeFinalResults(staticMap, trajectory, staticCounts, dynamicCounts, frameRange, opts);
        catch ME
            warning('Final visualization failed: %s', ME.message);
            if ~isempty(ME.stack)
                fprintf('  Error location: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
            end
        end
    else
        fprintf('[Visualization] Skipping Figure 2: insufficient data.\n');
        fprintf('  Map points: %d, Trajectory points: %d\n', ...
            size(staticMap.positions, 2), size(trajectory, 2));
    end
end

end

function visualizeFinalResults(staticMap, trajectory, staticCounts, dynamicCounts, frameRange, opts)
fprintf('\n[Visualization] ===== Starting Final Results Visualization =====\n');
fprintf('[Visualization] Map: %dx%d points\n', size(staticMap.positions, 1), size(staticMap.positions, 2));
fprintf('[Visualization] Trajectory: %dx%d points\n', size(trajectory, 1), size(trajectory, 2));
fprintf('[Visualization] Frame range: %d to %d\n', frameRange(1), frameRange(end));

% Validate data before creating figure
if isempty(staticMap.positions) || size(staticMap.positions, 2) == 0
    fprintf('[Visualization] ERROR: Map is empty, skipping Figure 2.\n');
    return;
end

if isempty(trajectory) || size(trajectory, 2) < 2
    fprintf('[Visualization] ERROR: Trajectory is invalid, skipping Figure 2.\n');
    return;
end

hasMapData = ~isempty(staticMap.positions) && size(staticMap.positions, 2) > 0;
hasTrajectoryData = size(trajectory, 2) >= 2;
hasCountData = ~isempty(frameRange) && ~isempty(staticCounts) && ~isempty(dynamicCounts) && ...
    numel(frameRange) == numel(staticCounts) && numel(frameRange) == numel(dynamicCounts);
lastStaticWorld = opts.LastStaticWorld;
lastDynamicWorld = opts.LastDynamicWorld;
hasStaticWorld = ~isempty(lastStaticWorld);
hasDynamicWorld = ~isempty(lastDynamicWorld);
hasConfidenceData = isfield(staticMap, 'confidence') && ~isempty(staticMap.confidence) && numel(staticMap.confidence) > 0;

if ~hasMapData && ~hasTrajectoryData && ~hasCountData && ~hasStaticWorld && ~hasDynamicWorld && ~hasConfidenceData
    fprintf('[Visualization] ERROR: No data to plot, skipping Figure 2.\n');
    return;
end

fig = figure('Name', 'StaticFusion Final Results', 'Position', [100, 100, 1400, 800], 'Visible', 'on');
clf(fig);
set(fig, 'Visible', 'on');
drawnow;
fprintf('[Visualization] Figure created.\n');

fprintf('[Visualization] Creating subplot 1 (3D map + trajectory)...\n');
try
    ax1 = subplot(2, 3, [1, 4]);
    hold(ax1, 'on');
    
    if ~isempty(staticMap.positions) && size(staticMap.positions, 2) > 0
        mapIdx = 1:opts.DownsampleMap:size(staticMap.positions, 2);
        mapIdx = mapIdx(mapIdx <= size(staticMap.positions, 2));
        
        if ~isempty(mapIdx)
            fprintf('[Visualization]   Plotting %d map points...\n', numel(mapIdx));
            if isfield(staticMap, 'colours') && ~isempty(staticMap.colours) && ...
                    size(staticMap.colours, 2) >= max(mapIdx)
                scatter3(ax1, staticMap.positions(1, mapIdx), ...
                    staticMap.positions(2, mapIdx), ...
                    staticMap.positions(3, mapIdx), ...
                    6, staticMap.colours(:, mapIdx)', '.');
            else
                scatter3(ax1, staticMap.positions(1, mapIdx), ...
                    staticMap.positions(2, mapIdx), ...
                    staticMap.positions(3, mapIdx), ...
                    6, 'b', '.');
            end
            fprintf('[Visualization]   Map points plotted.\n');
        else
            fprintf('[Visualization]   Warning: mapIdx is empty!\n');
        end
    else
        fprintf('[Visualization]   Warning: Map is empty!\n');
        text(ax1, 0.5, 0.5, 'No map data', 'HorizontalAlignment', 'center');
    end
    
    if size(trajectory, 2) >= 2
        fprintf('[Visualization]   Plotting trajectory (%d points)...\n', size(trajectory, 2));
        plot3(ax1, trajectory(1, :), trajectory(2, :), trajectory(3, :), ...
            'k-', 'LineWidth', 2, 'DisplayName', 'Camera Trajectory');
        scatter3(ax1, trajectory(1, 1), trajectory(2, 1), trajectory(3, 1), ...
            100, 'go', 'filled', 'DisplayName', 'Start');
        scatter3(ax1, trajectory(1, end), trajectory(2, end), trajectory(3, end), ...
            100, 'ro', 'filled', 'DisplayName', 'End');
        fprintf('[Visualization]   Trajectory plotted.\n');
    else
        fprintf('[Visualization]   Warning: Trajectory has < 2 points!\n');
    end
    
    if hasStaticWorld && size(lastStaticWorld, 2) > 0
        idxStatic = 1:opts.DownsampleFrame:size(lastStaticWorld, 2);
        hStatic = scatter3(ax1, lastStaticWorld(1, idxStatic), lastStaticWorld(2, idxStatic), lastStaticWorld(3, idxStatic), ...
            20, [0, 0.8, 0], 'filled', 'DisplayName', 'Final Static');
        set(hStatic, 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
    end
    if hasDynamicWorld && size(lastDynamicWorld, 2) > 0
        idxDynamic = 1:opts.DownsampleFrame:size(lastDynamicWorld, 2);
        hDynamic = scatter3(ax1, lastDynamicWorld(1, idxDynamic), lastDynamicWorld(2, idxDynamic), lastDynamicWorld(3, idxDynamic), ...
            20, [0.9, 0, 0], 'filled', 'DisplayName', 'Final Dynamic');
        set(hDynamic, 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
    end
    
    title(ax1, 'Static Background Map with Camera Trajectory');
    xlabel(ax1, 'X (m)'); ylabel(ax1, 'Y (m)'); zlabel(ax1, 'Z (m)');
    legend(ax1, 'Location', 'best');
    axis(ax1, 'equal'); grid(ax1, 'on');
    view(ax1, 135, 30);
    fprintf('[Visualization] Subplot 1 complete.\n');
catch ME
    fprintf('[Visualization] ERROR in subplot 1: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('  At %s line %d\n', ME.stack(1).file, ME.stack(1).line);
    end
end

fprintf('[Visualization] Creating subplot 2 (counts over time)...\n');
try
    ax2 = subplot(2, 3, 2);
    if ~isempty(frameRange) && ~isempty(staticCounts) && ~isempty(dynamicCounts) && ...
            numel(frameRange) == numel(staticCounts) && numel(frameRange) == numel(dynamicCounts)
        h1 = plot(ax2, frameRange, staticCounts, '-g', 'LineWidth', 2, 'DisplayName', 'Static'); hold(ax2, 'on');
        h2 = plot(ax2, frameRange, dynamicCounts, '-r', 'LineWidth', 2, 'DisplayName', 'Dynamic');
        xlabel(ax2, 'Frame'); ylabel(ax2, 'Point count (sampled)');
        title(ax2, 'Static vs Dynamic Points Over Time');
        legend(ax2, [h1, h2], {'Static', 'Dynamic'}, 'Location', 'best');
        grid(ax2, 'on');
        fprintf('[Visualization] Subplot 2 complete.\n');
        fprintf('[Visualization]   Static counts: min=%.0f, max=%.0f, mean=%.0f\n', ...
            min(staticCounts), max(staticCounts), mean(staticCounts));
        fprintf('[Visualization]   Dynamic counts: min=%.0f, max=%.0f, mean=%.0f\n', ...
            min(dynamicCounts), max(dynamicCounts), mean(dynamicCounts));
    else
        text(ax2, 0.5, 0.5, 'No count data', 'HorizontalAlignment', 'center');
        fprintf('[Visualization] Warning: Invalid count data for subplot 2.\n');
    end
catch ME
    fprintf('[Visualization] ERROR in subplot 2: %s\n', ME.message);
end

fprintf('[Visualization] Creating subplot 3 (confidence distribution)...\n');
try
    ax3 = subplot(2, 3, 3);
    if isfield(staticMap, 'confidence') && ~isempty(staticMap.confidence) && numel(staticMap.confidence) > 0
        histogram(ax3, staticMap.confidence, 30, 'FaceColor', 'blue', 'EdgeColor', 'black');
    else
        text(ax3, 0.5, 0.5, 'No confidence data', 'HorizontalAlignment', 'center');
    end
    xlabel(ax3, 'Confidence'); ylabel(ax3, 'Surfel count');
    title(ax3, 'Surfel Confidence Distribution');
    grid(ax3, 'on');
    fprintf('[Visualization] Subplot 3 complete.\n');
catch ME
    fprintf('[Visualization] ERROR in subplot 3: %s\n', ME.message);
end

fprintf('[Visualization] Creating subplot 4 (trajectory + segmentation top view)...\n');
try
    ax4 = subplot(2, 3, 5);
    hold(ax4, 'on');
    if hasDynamicWorld && size(lastDynamicWorld, 2) > 0
        idxDynamic2 = 1:opts.DownsampleFrame:size(lastDynamicWorld, 2);
        scatter(ax4, lastDynamicWorld(1, idxDynamic2), lastDynamicWorld(2, idxDynamic2), ...
            15, [0.9, 0, 0], 'filled', 'DisplayName', 'Dynamic');
    end
    if hasStaticWorld && size(lastStaticWorld, 2) > 0
        idxStatic2 = 1:opts.DownsampleFrame:size(lastStaticWorld, 2);
        scatter(ax4, lastStaticWorld(1, idxStatic2), lastStaticWorld(2, idxStatic2), ...
            15, [0, 0.8, 0], 'filled', 'DisplayName', 'Static');
    end
    if size(trajectory, 2) >= 2
        plot(ax4, trajectory(1, :), trajectory(2, :), 'b-', 'LineWidth', 2, 'DisplayName', 'Trajectory');
        scatter(ax4, trajectory(1, 1), trajectory(2, 1), 80, 'go', 'filled');
        scatter(ax4, trajectory(1, end), trajectory(2, end), 80, 'ro', 'filled');
    end
    xlabel(ax4, 'X (m)'); ylabel(ax4, 'Y (m)');
    title(ax4, 'Final Frame Segmentation (Top View)');
    axis(ax4, 'equal'); grid(ax4, 'on');
    legend(ax4, 'Location', 'bestoutside');
    fprintf('[Visualization] Subplot 4 complete.\n');
catch ME
    fprintf('[Visualization] ERROR in subplot 4: %s\n', ME.message);
end

fprintf('[Visualization] Creating subplot 5 (statistics)...\n');
try
    ax5 = subplot(2, 3, 6);
    axis(ax5, 'off');
    
    avgStatic = mean(staticCounts);
    avgDynamic = mean(dynamicCounts);
    if size(trajectory, 2) > 1
        trajDiff = diff(trajectory, 1, 2);
        trajLength = sum(sqrt(sum(trajDiff.^2, 1)));
    else
        trajLength = 0;
    end
    
    finalStaticCount = hasStaticWorld * size(lastStaticWorld, 2);
    finalDynamicCount = hasDynamicWorld * size(lastDynamicWorld, 2);
    
    if isfield(staticMap, 'confidence') && ~isempty(staticMap.confidence) && numel(staticMap.confidence) > 0
        maxConf = max(staticMap.confidence);
        minConf = min(staticMap.confidence);
        meanConf = mean(staticMap.confidence);
        hasConf = true;
    else
        hasConf = false;
    end
    
    totalSum = avgStatic + avgDynamic;
    if totalSum > 0
        staticRatio = 100 * avgStatic / totalSum;
    else
        staticRatio = 0;
    end
    
    statsText = {
        sprintf('Total Frames: %d', numel(frameRange));
        sprintf('Total Surfels: %d', size(staticMap.positions, 2));
        sprintf('Avg Static Points: %.0f', avgStatic);
        sprintf('Avg Dynamic Points: %.0f', avgDynamic);
        sprintf('Static Ratio: %.1f%%', staticRatio);
    };
    
    if hasConf
        statsText{end+1} = sprintf('Max Confidence: %.2f', maxConf);
        statsText{end+1} = sprintf('Min Confidence: %.2f', minConf);
        statsText{end+1} = sprintf('Mean Confidence: %.2f', meanConf);
    end
    
    statsText{end+1} = '';
    statsText{end+1} = sprintf('Trajectory Length: %.2f m', trajLength);
    if ~isempty(opts.LastFrameID)
        statsText{end+1} = sprintf('Final Frame ID: %d', opts.LastFrameID);
    end
    statsText{end+1} = sprintf('Final Static Points: %d', finalStaticCount);
    statsText{end+1} = sprintf('Final Dynamic Points: %d', finalDynamicCount);
    if (finalStaticCount + finalDynamicCount) > 0
        statsText{end+1} = sprintf('Final Static Ratio: %.1f%%', ...
            100 * finalStaticCount / (finalStaticCount + finalDynamicCount));
    end
    
    text(ax5, 0.1, 0.5, statsText, 'FontSize', 11, ...
        'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left');
    title(ax5, 'Statistics Summary', 'FontSize', 12, 'FontWeight', 'bold');
    fprintf('[Visualization] Subplot 5 complete.\n');
catch ME
    fprintf('[Visualization] ERROR in subplot 5: %s\n', ME.message);
    text(ax5, 0.5, 0.5, sprintf('Error: %s', ME.message), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

try
    sgtitle(fig, 'StaticFusion Reconstruction Results', 'FontSize', 14, 'FontWeight', 'bold');
catch
    annotation(fig, 'textbox', [0.3, 0.95, 0.4, 0.05], ...
        'String', 'StaticFusion Reconstruction Results', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'EdgeColor', 'none');
end

set(fig, 'Visible', 'on');
drawnow;
pause(0.1);
refresh(fig);
drawnow;

try
    children = get(fig, 'Children');
    hasContent = false;
    for i = 1:numel(children)
        try
            if isa(children(i), 'matlab.graphics.axis.Axes')
                ax = children(i);
                axChildren = get(ax, 'Children');
                for j = 1:numel(axChildren)
                    obj = axChildren(j);
                    if ~isa(obj, 'matlab.graphics.primitive.Text') && ...
                       ~isa(obj, 'matlab.graphics.primitive.Legend')
                        hasContent = true;
                        break;
                    end
                end
                if hasContent
                    break;
                end
            end
        catch
            continue;
        end
    end
    
    if ~hasContent
        fprintf('[Visualization] WARNING: Figure 2 created but has no plot content. Closing empty figure.\n');
        close(fig);
        return;
    else
        fprintf('[Visualization] ===== Final Results Visualization Complete =====\n\n');
    end
catch ME
    fprintf('[Visualization] Warning: Could not verify figure content: %s\n', ME.message);
    fprintf('[Visualization] ===== Final Results Visualization Complete =====\n\n');
end
end
