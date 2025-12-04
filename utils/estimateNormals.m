function normals = estimateNormals(points, varargin)
%ESTIMATENORMALS Estimate surface normals for an unstructured point cloud.
%   normals = ESTIMATENORMALS(points) computes per-point normals for a
%   3xN array of XYZ coordinates (metres) using PCA on the K nearest
%   neighbours.
%
%   Name-value pairs:
%       'K'         - number of neighbours (default 20, must be >=3)
%       'ViewPoint' - 3x1 vector; normals are oriented to face this point
%                     (optional). Use [] to skip orientation.
%
%   The returned normals are a 3xN array with unit length columns.

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

% Validate input
if any(isnan(pts(:))) || any(isinf(pts(:)))
    warning('estimateNormals:InvalidPoints', 'Input contains NaN or Inf values.');
    normals = zeros(3, numPoints);
    normals(3, :) = 1;  % Default to [0,0,1]
    return;
end

% Nearest neighbours (including the point itself).
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
    % Validate indices
    neighbourIdx = neighbourIdx(neighbourIdx >= 1 & neighbourIdx <= numPoints);
    if numel(neighbourIdx) < 3
        normals(i, :) = [0, 0, 1];  % Default normal
        continue;
    end
    neighbours = pts(neighbourIdx, :);
    centroid = mean(neighbours, 1);
    centred = neighbours - centroid;

    C = centred' * centred;
    if any(isnan(C(:))) || any(isinf(C(:)))
        normals(i, :) = [0, 0, 1];  % Default normal
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

