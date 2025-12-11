function plotStaticMapStructure(map, varargin)
% Visualize surfel-based static map structure
narginchk(1, inf);

validateattributes(map, {'struct'}, {'scalar'}, mfilename, 'map', 1);

parser = inputParser;
parser.addParameter('Downsample', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('Title', 'Static Map Structure', @(x) isstring(x) || ischar(x));
parser.addParameter('FigureHandle', [], @(x) isempty(x) || ishghandle(x));
parser.parse(varargin{:});
opts = parser.Results;

titleStr = char(opts.Title);
downsample = max(1, round(opts.Downsample));

hasPositions = isfield(map, 'positions') && ~isempty(map.positions);
if ~hasPositions
    error('plotStaticMapStructure:MissingPositions', 'Map.positions is required and cannot be empty.');
end

positions = map.positions;
numSurfels = size(positions, 2);

hasColours = isfield(map, 'colours') && ~isempty(map.colours) && size(map.colours, 2) == numSurfels;
hasNormals = isfield(map, 'normals') && ~isempty(map.normals) && size(map.normals, 2) == numSurfels;

sampleIdx = 1:downsample:numSurfels;
if isempty(sampleIdx)
    sampleIdx = 1:numSurfels;
end

fprintf('\n[plotStaticMapStructure] %s\n', titleStr);
fprintf('  Surfels              : %d (displaying %d)\n', numSurfels, numel(sampleIdx));
fprintf('  Downsample step      : %d\n', downsample);
if isfield(map, 'confidence') && ~isempty(map.confidence)
    fprintf('  Confidence range     : [%.2f, %.2f]\n', min(map.confidence), max(map.confidence));
end
if isfield(map, 'radius') && ~isempty(map.radius)
    fprintf('  Radius range (m)     : [%.3f, %.3f]\n', min(map.radius), max(map.radius));
end
posMin = min(positions, [], 2);
posMax = max(positions, [], 2);
fprintf('  Position bounds (XYZ):\n');
fprintf('    X: [%.3f, %.3f] m\n', posMin(1), posMax(1));
fprintf('    Y: [%.3f, %.3f] m\n', posMin(2), posMax(2));
    fprintf('    Z: [%.3f, %.3f] m\n\n', posMin(3), posMax(3));

if isempty(opts.FigureHandle)
    fig = figure('Name', titleStr, 'Color', 'w', 'Position', [100, 100, 1200, 600]);
else
    fig = opts.FigureHandle;
    figure(fig); clf(fig);
end

tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
sgtitle(fig, titleStr, 'FontSize', 14, 'FontWeight', 'bold');

nexttile;
hold on;
if hasColours
    scatter3(positions(1, sampleIdx), positions(2, sampleIdx), positions(3, sampleIdx), ...
        10, map.colours(:, sampleIdx)', '.');
else
    scatter3(positions(1, sampleIdx), positions(2, sampleIdx), positions(3, sampleIdx), ...
        10, 'b', '.');
end
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Surfels (displayed %d / %d)', numel(sampleIdx), numSurfels));
view(135, 30);

nexttile;
if hasNormals
    quiver3(positions(1, sampleIdx), positions(2, sampleIdx), positions(3, sampleIdx), ...
        map.normals(1, sampleIdx), map.normals(2, sampleIdx), map.normals(3, sampleIdx), 0.2, 'k');
    axis equal; grid on;
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('Normals (scaled)');
    view(135, 30);
else
    axis off;
    text(0.5, 0.5, 'No normals available', 'HorizontalAlignment', 'center');
end

end
