function map = cleanupStaticMap(map, varargin)
% Remove low-confidence surfels and merge duplicates

narginchk(1, inf);
validateStaticMap(map);

parser = inputParser;
addParameter(parser, 'MinConfidence', map.params.minConfidence, ...
    @(x) validateattributes(x, {'double'}, {'scalar', 'nonnegative'}));
addParameter(parser, 'MergeRadius', map.params.mergeRadius, ...
    @(x) validateattributes(x, {'double'}, {'scalar', 'positive'}));
addParameter(parser, 'NormalSimilarity', 0.9, ...
    @(x) validateattributes(x, {'double'}, {'scalar', '>=', -1, '<=', 1}));
addParameter(parser, 'Verbose', false, @(x) islogical(x) && isscalar(x));
parse(parser, varargin{:});
opts = parser.Results;

if isempty(map.positions) || size(map.positions, 2) == 0
    if opts.Verbose
        fprintf('[cleanupStaticMap] Map is empty, nothing to clean.\n');
    end
    return;
end

initialCount = size(map.positions, 2);
if opts.Verbose
    fprintf('[cleanupStaticMap] Starting cleanup: %d surfels.\n', initialCount);
end

confidenceMask = map.confidence >= opts.MinConfidence;
removedLowConf = nnz(~confidenceMask);

if opts.Verbose && removedLowConf > 0
    fprintf('[cleanupStaticMap] Removing %d low-confidence surfels (confidence < %.2f).\n', ...
        removedLowConf, opts.MinConfidence);
end

map.positions = map.positions(:, confidenceMask);
map.normals = map.normals(:, confidenceMask);
map.colours = map.colours(:, confidenceMask);
map.confidence = map.confidence(confidenceMask);
map.radius = map.radius(confidenceMask);

if isempty(map.positions)
    if opts.Verbose
        fprintf('[cleanupStaticMap] All surfels removed. Map is now empty.\n');
    end
    return;
end

if opts.Verbose
    fprintf('[cleanupStaticMap] Merging duplicate surfels (radius=%.3f, normal similarity=%.2f)...\n', ...
        opts.MergeRadius, opts.NormalSimilarity);
end

numSurfels = size(map.positions, 2);
keepMask = true(1, numSurfels);
mergedCount = 0;

if numSurfels > 50000
    if opts.Verbose
        fprintf('[cleanupStaticMap] Very large map detected (%d surfels). Skipping merge step for performance.\n', ...
            numSurfels);
        fprintf('[cleanupStaticMap] Only low-confidence surfels were removed in step 1.\n');
    end
    finalCount = size(map.positions, 2);
    if opts.Verbose
        fprintf('[cleanupStaticMap] Cleanup complete: %d -> %d surfels (removed %d low-confidence).\n', ...
            initialCount, finalCount, initialCount - finalCount);
    end
    return;
end

if numSurfels > 20000
    MAX_PROCESS = 5000;
elseif numSurfels > 10000
    MAX_PROCESS = 10000;
else
    MAX_PROCESS = numSurfels;
end

if numSurfels > MAX_PROCESS
    if opts.Verbose
        fprintf('[cleanupStaticMap] Large map detected (%d surfels). Processing %d surfels per pass.\n', ...
            numSurfels, MAX_PROCESS);
    end
    if opts.Verbose
        fprintf('[cleanupStaticMap] Sorting surfels by confidence...\n');
    end
    [~, sortIdx] = sort(map.confidence, 'ascend');
    processIndices = sortIdx(1:MAX_PROCESS);
else
    processIndices = 1:numSurfels;
end

if opts.Verbose
    fprintf('[cleanupStaticMap] Building spatial index...\n');
end

batchSize = 200;
numBatches = ceil(length(processIndices) / batchSize);

for batchIdx = 1:numBatches
    batchStart = (batchIdx - 1) * batchSize + 1;
    batchEnd = min(batchIdx * batchSize, length(processIndices));
    batchIndices = processIndices(batchStart:batchEnd);
    
    if opts.Verbose && (mod(batchIdx, 5) == 0 || batchIdx == 1)
        fprintf('[cleanupStaticMap] Processing batch %d/%d (processed %d/%d surfels)...\n', ...
            batchIdx, numBatches, batchEnd, length(processIndices));
    end
    
    for i = batchIndices
        if ~keepMask(i)
            continue;
        end
        
        if mod(i - batchIndices(1), 50) == 0 || i == batchIndices(1)
            remainingIndices = find(keepMask);
            if length(remainingIndices) <= 1
                break;
            end
            remainingPositions = map.positions(:, remainingIndices);
        end
        
        if length(remainingIndices) <= 1
            break;
        end
        
        try
            [indices, distances] = rangesearch(remainingPositions', ...
                map.positions(:, i)', opts.MergeRadius);
            if isempty(indices) || isempty(indices{1})
                continue;
            end
            candidateIndices = remainingIndices(indices{1});
            distances = distances{1};
            if length(candidateIndices) > 30
                [sortedDist, sortIdx] = sort(distances);
                candidateIndices = candidateIndices(sortIdx(1:30));
                distances = sortedDist(1:30);
            end
        catch
            try
                [indices, distances] = knnsearch(remainingPositions', ...
                    map.positions(:, i)', 'K', min(20, length(remainingIndices)));
                candidateIndices = remainingIndices(indices);
            catch
                continue;
            end
        end
        
        validMask = distances <= opts.MergeRadius & distances > 0;
        validCandidates = candidateIndices(validMask);
        if isempty(validCandidates)
            continue;
        end
        
        normalI = map.normals(:, i);
        normalsJ = map.normals(:, validCandidates);
        cosSim = sum(normalI .* normalsJ, 1);
        similarMask = cosSim >= opts.NormalSimilarity;
        mergeCandidates = validCandidates(similarMask);
        
        mergeCandidates = mergeCandidates(keepMask(mergeCandidates));
        
        if isempty(mergeCandidates)
            continue;
        end
        
        totalConf = map.confidence(i) + sum(map.confidence(mergeCandidates));
        if totalConf > 0
            alpha = map.confidence(i) / totalConf;
            
            map.positions(:, i) = alpha * map.positions(:, i) + ...
                (1 - alpha) * mean(map.positions(:, mergeCandidates), 2);
            
            mergedNormal = alpha * map.normals(:, i) + ...
                (1 - alpha) * mean(map.normals(:, mergeCandidates), 2);
            normVal = norm(mergedNormal);
            if normVal > eps
                map.normals(:, i) = mergedNormal / normVal;
            end
            
            map.colours(:, i) = alpha * map.colours(:, i) + ...
                (1 - alpha) * mean(map.colours(:, mergeCandidates), 2);
            
            map.confidence(i) = min(map.params.maxConfidence, totalConf);
            map.radius(i) = max(map.radius(i), max(map.radius(mergeCandidates)));
        end
        
        keepMask(mergeCandidates) = false;
        mergedCount = mergedCount + length(mergeCandidates);
    end
end

if mergedCount > 0
    if opts.Verbose
        fprintf('[cleanupStaticMap] Merged %d surfels into %d groups.\n', ...
            mergedCount, nnz(keepMask));
    end
    
    map.positions = map.positions(:, keepMask);
    map.normals = map.normals(:, keepMask);
    map.colours = map.colours(:, keepMask);
    map.confidence = map.confidence(keepMask);
    map.radius = map.radius(keepMask);
end

finalCount = size(map.positions, 2);
if opts.Verbose
    fprintf('[cleanupStaticMap] Cleanup complete: %d -> %d surfels (removed %d).\n', ...
        initialCount, finalCount, initialCount - finalCount);
end

end

function validateStaticMap(map)
if ~isstruct(map)
    error('Static map must be a struct.');
end
requiredFields = {'positions', 'normals', 'colours', 'confidence', 'radius', 'params'};
for f = requiredFields
    if ~isfield(map, f{1})
        error('cleanupStaticMap:MissingField', ...
            'Static map struct missing field ''%s''.', f{1});
    end
end
end

