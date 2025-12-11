function [keepMask, labels] = filterClusters(points, varargin)
% Keep clusters that exceed minimum size using radius-based region growing

arguments
    points double
end

p = inputParser;
addParameter(p, 'radius', 0.1);
addParameter(p, 'minClusterSize', 30);
parse(p, varargin{:});
opts = p.Results;

if isempty(points)
    keepMask = false(size(points, 2), 1);
    labels = zeros(0, 1);
    return;
end

if size(points, 1) == 3
    pts = points';
else
    pts = points;
end

[neighborCells, ~] = rangesearch(pts, pts, opts.radius);
numPts = size(pts, 1);
labels = zeros(numPts, 1);
clusterId = 0;

for i = 1:numPts
    if labels(i) ~= 0
        continue;
    end
    clusterId = clusterId + 1;
    queue = i;
    labels(i) = clusterId;
    head = 1;

    while head <= numel(queue)
        current = queue(head);
        neighbours = neighborCells{current};
        for j = neighbours
            if labels(j) == 0
                labels(j) = clusterId;
                queue(end + 1) = j; %#ok<AGROW>
            end
        end
        head = head + 1;
    end
end

clusterCounts = accumarray(labels, 1);
validClusters = clusterCounts >= opts.minClusterSize;
keepMask = validClusters(labels);

if size(points, 1) == 3
    keepMask = keepMask(:)';
else
    keepMask = keepMask;
end

end

