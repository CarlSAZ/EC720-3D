function normals = estimateNormals(points, varargin)
% Estimate surface normals using PCA on K nearest neighbours

narginchk(1, inf);
validateattributes(points, {'double'}, {'size',[3, NaN]}, mfilename, 'points', 1);

parser = inputParser;
addParameter(parser, 'K', 20, @(x) validateattributes(x, {'double'}, {'scalar','integer','>=',3}));
addParameter(parser, 'ViewPoint', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 3));
parse(parser, varargin{:});
opts = parser.Results;

if isempty(points)
    normals = zeros(3, 0);
    return;
end

pts = points';
numPoints = size(pts, 1);
K = min(opts.K, numPoints);

if any(isnan(pts(:))) || any(isinf(pts(:)))
    warning('estimateNormals:InvalidPoints', 'Input contains NaN or Inf values.');
    normals = zeros(3, numPoints);
    normals(3, :) = 1;  % Default to [0,0,1]
    return;
end

try
    [indices, ~] = knnsearch(pts, pts, 'K', K);
catch ME
    warning('estimateNormals:KnnSearchFailed', 'knnsearch failed: %s', ME.message);
    normals = zeros(3, numPoints);
    normals(3, :) = 1;  % Default to [0,0,1]
    return;
end

normals = zeros(numPoints, 3);

for i = 1:numPoints
    neighbourIdx = indices(i, :);
    neighbourIdx = neighbourIdx(neighbourIdx >= 1 & neighbourIdx <= numPoints);
    if numel(neighbourIdx) < 3
        normals(i, :) = [0, 0, 1];
        continue;
    end
    neighbours = pts(neighbourIdx, :);
    centroid = mean(neighbours, 1);
    centred = neighbours - centroid;

    C = centred' * centred;
    if any(isnan(C(:))) || any(isinf(C(:)))
        normals(i, :) = [0, 0, 1];
        continue;
    end
    [V, D] = eig(C);

    [~, minIdx] = min(diag(D));
    n = V(:, minIdx);

    if ~isempty(opts.ViewPoint)
        viewVec = opts.ViewPoint(:)' - pts(i, :);
        if dot(n, viewVec) > 0
            n = -n;
        end
    end

    normals(i, :) = n / max(norm(n), eps);
end

normals = normals';
end

