function visualizeChanges(pcBase, pcOverlay, addedXYZ, removedXYZ, varargin)
% Render base point cloud with added/removed points

narginchk(4, inf);
validateattributes(pcBase, {'struct'}, {}, mfilename, 'pcBase', 1);
validateattributes(pcOverlay, {'struct'}, {}, mfilename, 'pcOverlay', 2);
validateattributes(addedXYZ, {'double'}, {}, mfilename, 'addedXYZ', 3);
validateattributes(removedXYZ, {'double'}, {}, mfilename, 'removedXYZ', 4);

p = inputParser;
addParameter(p, 'downsample', 5);
addParameter(p, 'views', {[135, 30]});
addParameter(p, 'showLegend', true);
addParameter(p, 'markerSize', 12);
addParameter(p, 'savePath', "");
parse(p, varargin{:});
opts = p.Results;

if isempty(pcBase.XYZ)
    warning('Base point cloud is empty; skipping visualisation.');
    return;
end

baseIdx = 1:opts.downsample:size(pcBase.XYZ, 2);
overlayIdx = 1:opts.downsample:size(pcOverlay.XYZ, 2);

for v = 1:numel(opts.views)
    figure;
    hold on; grid on; axis equal;

    scatter3(pcOverlay.XYZ(1, overlayIdx), pcOverlay.XYZ(2, overlayIdx), ...
        pcOverlay.XYZ(3, overlayIdx), 3, pcOverlay.RGB(:, overlayIdx)', '.');

    legendEntries = {'Overlay'};

    if ~isempty(addedXYZ)
        scatter3(addedXYZ(1, :), addedXYZ(2, :), addedXYZ(3, :), ...
            opts.markerSize, 'r', 'filled');
        legendEntries{end + 1} = 'Added';
    end
    if ~isempty(removedXYZ)
        scatter3(removedXYZ(1, :), removedXYZ(2, :), removedXYZ(3, :), ...
            opts.markerSize, 'b', 'filled');
        legendEntries{end + 1} = 'Removed';
    end

    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    view(opts.views{v});

    if opts.showLegend
        legend(legendEntries, 'Location', 'best');
    end

    title(sprintf('Change Visualisation (view %d)', v));

    if opts.savePath ~= ""
        filename = fullfile(opts.savePath, sprintf('change_view_%d.png', v));
        exportgraphics(gcf, filename);
    end
end

end

