function diff = computePointCloudDiff(pcA, pcB, varargin)
% Detect changes between two point clouds using nearest-neighbour distances

narginchk(2, inf);
validateattributes(pcA, {'struct'}, {}, mfilename, 'pcA', 1);
validateattributes(pcB, {'struct'}, {}, mfilename, 'pcB', 2);

p = inputParser;
addParameter(p, 'threshold', 0.05);
addParameter(p, 'requireMutual', true);
parse(p, varargin{:});
params = p.Results;

diff = struct();

if ~isfield(pcA, 'XYZ') || ~isfield(pcB, 'XYZ')
    error('Point cloud structs must contain an ''XYZ'' field.');
end

if isempty(pcA.XYZ) || isempty(pcB.XYZ)
    diff.distAB = [];
    diff.idxAB = [];
    diff.distBA = [];
    diff.idxBA = [];
    diff.addedMask = false(1, size(pcB.XYZ, 2));
    diff.removedMask = false(1, size(pcA.XYZ, 2));
    diff.addedXYZ = zeros(3, 0);
    diff.removedXYZ = zeros(3, 0);
    return;
end

[diff.idxAB, diff.distAB] = knnsearch(pcA.XYZ', pcB.XYZ');
[diff.idxBA, diff.distBA] = knnsearch(pcB.XYZ', pcA.XYZ');

addedMask = diff.distAB > params.threshold;
removedMask = diff.distBA > params.threshold;

if params.requireMutual
    mutualAdded = false(size(addedMask));
    for i = find(addedMask)
        neighbourIdx = diff.idxAB(i);
        if diff.distBA(neighbourIdx) > params.threshold
            mutualAdded(i) = true;
        end
    end
    addedMask = mutualAdded;

    mutualRemoved = false(size(removedMask));
    for i = find(removedMask)
        neighbourIdx = diff.idxBA(i);
        if diff.distAB(neighbourIdx) > params.threshold
            mutualRemoved(i) = true;
        end
    end
    removedMask = mutualRemoved;
end

diff.addedMask = addedMask;
diff.removedMask = removedMask;
diff.addedXYZ = pcB.XYZ(:, addedMask);
diff.removedXYZ = pcA.XYZ(:, removedMask);

end

