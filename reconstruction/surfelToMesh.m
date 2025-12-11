function mesh = surfelToMesh(staticMap, varargin)
% Convert surfel-based static map to triangular mesh

narginchk(1, inf);
validateStaticMap(staticMap);

parser = inputParser;
addParameter(parser, 'Method', 'delaunay', @(x) ismember(x, {'delaunay', 'poisson'}));
addParameter(parser, 'MaxDistance', 0.1, @(x) validateattributes(x, {'double'}, {'scalar', 'positive'}));
addParameter(parser, 'NormalWeight', 0.8, @(x) validateattributes(x, {'double'}, {'scalar', '>=', 0, '<=', 1}));
addParameter(parser, 'Verbose', false, @(x) islogical(x) && isscalar(x));
parse(parser, varargin{:});
opts = parser.Results;

if isempty(staticMap.positions) || size(staticMap.positions, 2) == 0
    error('surfelToMesh:EmptyMap', 'Static map is empty.');
end

if opts.Verbose
    fprintf('[surfelToMesh] Converting %d surfels to mesh using %s method...\n', ...
        size(staticMap.positions, 2), opts.Method);
end

switch lower(opts.Method)
    case 'delaunay'
        mesh = delaunayMesh(staticMap, opts);
    case 'poisson'
        if opts.Verbose
            warning('[surfelToMesh] Poisson method not available, using Delaunay.');
        end
        mesh = delaunayMesh(staticMap, opts);
    otherwise
        error('surfelToMesh:UnknownMethod', 'Unknown method: %s', opts.Method);
end

if opts.Verbose
    fprintf('[surfelToMesh] Mesh generated: %d vertices, %d faces.\n', ...
        size(mesh.vertices, 2), size(mesh.faces, 2));
end

end

function mesh = delaunayMesh(staticMap, opts)
points = staticMap.positions';
normals = staticMap.normals';
colors = staticMap.colours';

numPoints = size(points, 1);
if opts.Verbose
    fprintf('[surfelToMesh] Computing Delaunay triangulation for %d points...\n', numPoints);
end

try
    dt = delaunayTriangulation(points);
    faces = dt.ConnectivityList;
    
    if opts.Verbose
        fprintf('[surfelToMesh] Filtering %d initial faces...\n', size(faces, 1));
    end
    
    validFaces = true(size(faces, 1), 1);
    for i = 1:size(faces, 1)
        face = faces(i, :);
        v1 = points(face(1), :)';
        v2 = points(face(2), :)';
        v3 = points(face(3), :)';
        
        edge1 = norm(v2 - v1);
        edge2 = norm(v3 - v2);
        edge3 = norm(v1 - v3);
        
        if max([edge1, edge2, edge3]) > opts.MaxDistance
            validFaces(i) = false;
            continue;
        end
        
        if opts.NormalWeight > 0
            faceNormal = cross(v2 - v1, v3 - v1);
            faceNormal = faceNormal / (norm(faceNormal) + eps);
            
            n1 = normals(face(1), :)';
            n2 = normals(face(2), :)';
            n3 = normals(face(3), :)';
            avgNormal = (n1 + n2 + n3) / 3;
            avgNormal = avgNormal / (norm(avgNormal) + eps);
            
            normalSim = dot(faceNormal, avgNormal);
            if normalSim < opts.NormalWeight
                validFaces(i) = false;
            end
        end
    end
    
    faces = faces(validFaces, :);
    
catch ME
    if opts.Verbose
        warning('[surfelToMesh] Delaunay triangulation failed: %s. Using simplified approach.', ME.message);
    end
    faces = createSimpleMesh(points, normals, opts);
end

mesh = struct();
mesh.vertices = points';
mesh.faces = faces';
mesh.normals = normals';
mesh.colors = colors';

end

function faces = createSimpleMesh(points, normals, opts)
numPoints = size(points, 1);
faces = zeros(0, 3);
faceCount = 0;

for i = 1:min(1000, numPoints)
    distances = sqrt(sum((points - points(i, :)).^2, 2));
    neighbors = find(distances > 0 & distances <= opts.MaxDistance);
    
    if length(neighbors) < 2
        continue;
    end
    
    for j = 1:min(5, length(neighbors)-1)
        for k = (j+1):min(5, length(neighbors))
            n1 = neighbors(j);
            n2 = neighbors(k);
            
            if opts.NormalWeight > 0
                n_i = normals(i, :)';
                n_j = normals(n1, :)';
                n_k = normals(n2, :)';
                avgNormal = (n_i + n_j + n_k) / 3;
                avgNormal = avgNormal / (norm(avgNormal) + eps);
                
                faceNormal = cross(points(n1, :) - points(i, :), points(n2, :) - points(i, :));
                faceNormal = faceNormal / (norm(faceNormal) + eps);
                
                if dot(faceNormal, avgNormal) < opts.NormalWeight
                    continue;
                end
            end
            
            faceCount = faceCount + 1;
            if faceCount > size(faces, 1)
                faces = [faces; zeros(1000, 3)];
            end
            faces(faceCount, :) = [i, n1, n2];
        end
    end
end

faces = faces(1:faceCount, :);
end

function validateStaticMap(map)
if ~isstruct(map)
    error('Static map must be a struct.');
end
requiredFields = {'positions', 'normals', 'colours', 'confidence', 'radius', 'params'};
for f = requiredFields
    if ~isfield(map, f{1})
        error('surfelToMesh:MissingField', ...
            'Static map struct missing field ''%s''.', f{1});
    end
end
end

