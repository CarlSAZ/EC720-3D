function exportMesh(mesh, filename, varargin)
%EXPORTMESH Export mesh to PLY or OBJ format.
%   EXPORTMESH(mesh, filename) exports the mesh structure to a file.
%   The format is determined by the file extension (.ply or .obj).
%
%   Input:
%       mesh        - struct with fields vertices, faces, normals, colors
%       filename    - output file path
%
%   Name-value pairs:
%       'Format'    - 'auto' (default, from extension), 'ply', or 'obj'
%       'IncludeNormals' - include vertex normals (default: true)
%       'IncludeColors'  - include vertex colors (default: true)
%
%   Mesh struct fields:
%       vertices    - 3xN vertex positions
%       faces       - 3xM face indices (1-indexed)
%       normals     - 3xN vertex normals (optional)
%       colors      - 3xN vertex colors in [0,1] (optional)

narginchk(2, inf);

parser = inputParser;
addParameter(parser, 'Format', 'auto', @(x) ismember(x, {'auto', 'ply', 'obj'}));
addParameter(parser, 'IncludeNormals', true, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'IncludeColors', true, @(x) islogical(x) && isscalar(x));
parse(parser, varargin{:});
opts = parser.Results;

% Validate mesh structure
if ~isstruct(mesh)
    error('exportMesh:InvalidMesh', 'Mesh must be a struct.');
end
if ~isfield(mesh, 'vertices') || ~isfield(mesh, 'faces')
    error('exportMesh:MissingFields', 'Mesh must have vertices and faces fields.');
end

vertices = mesh.vertices;
faces = mesh.faces;

if size(vertices, 1) ~= 3
    error('exportMesh:InvalidVertices', 'Vertices must be 3xN.');
end
if size(faces, 1) ~= 3
    error('exportMesh:InvalidFaces', 'Faces must be 3xM.');
end

% Determine format from extension if auto
if strcmp(opts.Format, 'auto')
    [~, ~, ext] = fileparts(filename);
    if strcmpi(ext, '.ply')
        format = 'ply';
    elseif strcmpi(ext, '.obj')
        format = 'obj';
    else
        % Default to PLY if no extension
        format = 'ply';
        filename = [filename, '.ply'];
    end
else
    format = lower(opts.Format);
end

% Get optional fields
hasNormals = opts.IncludeNormals && isfield(mesh, 'normals') && ...
    size(mesh.normals, 1) == 3 && size(mesh.normals, 2) == size(vertices, 2);
hasColors = opts.IncludeColors && isfield(mesh, 'colors') && ...
    size(mesh.colors, 1) == 3 && size(mesh.colors, 2) == size(vertices, 2);

if hasNormals
    normals = mesh.normals;
else
    normals = [];
end

if hasColors
    colors = mesh.colors;
    % Ensure colors are in [0, 255] range for export
    if max(colors(:)) <= 1.0
        colors = colors * 255;
    end
    colors = round(colors);
    colors = max(0, min(255, colors));
else
    colors = [];
end

% Export based on format
switch format
    case 'ply'
        exportPLY(filename, vertices, faces, normals, colors);
    case 'obj'
        exportOBJ(filename, vertices, faces, normals, colors);
    otherwise
        error('exportMesh:UnknownFormat', 'Unknown format: %s', format);
end

fprintf('[exportMesh] Mesh exported to %s (%d vertices, %d faces).\n', ...
    filename, size(vertices, 2), size(faces, 2));

end

%% Local functions
function exportPLY(filename, vertices, faces, normals, colors)
% Export to PLY format

fid = fopen(filename, 'w');
if fid == -1
    error('exportMesh:CannotOpenFile', 'Cannot open file: %s', filename);
end

numVertices = size(vertices, 2);
numFaces = size(faces, 2);
hasNormals = ~isempty(normals);
hasColors = ~isempty(colors);

% Write PLY header
fprintf(fid, 'ply\n');
fprintf(fid, 'format ascii 1.0\n');
fprintf(fid, 'comment Exported from StaticFusion\n');
fprintf(fid, 'element vertex %d\n', numVertices);
fprintf(fid, 'property float x\n');
fprintf(fid, 'property float y\n');
fprintf(fid, 'property float z\n');
if hasNormals
    fprintf(fid, 'property float nx\n');
    fprintf(fid, 'property float ny\n');
    fprintf(fid, 'property float nz\n');
end
if hasColors
    fprintf(fid, 'property uchar red\n');
    fprintf(fid, 'property uchar green\n');
    fprintf(fid, 'property uchar blue\n');
end
fprintf(fid, 'element face %d\n', numFaces);
fprintf(fid, 'property list uchar int vertex_indices\n');
fprintf(fid, 'end_header\n');

% Write vertices
for i = 1:numVertices
    fprintf(fid, '%.6f %.6f %.6f', vertices(1, i), vertices(2, i), vertices(3, i));
    if hasNormals
        fprintf(fid, ' %.6f %.6f %.6f', normals(1, i), normals(2, i), normals(3, i));
    end
    if hasColors
        fprintf(fid, ' %d %d %d', colors(1, i), colors(2, i), colors(3, i));
    end
    fprintf(fid, '\n');
end

% Write faces (PLY uses 0-indexed, but we have 1-indexed)
for i = 1:numFaces
    fprintf(fid, '3 %d %d %d\n', faces(1, i) - 1, faces(2, i) - 1, faces(3, i) - 1);
end

fclose(fid);

end

function exportOBJ(filename, vertices, faces, normals, colors)
% Export to OBJ format

fid = fopen(filename, 'w');
if fid == -1
    error('exportMesh:CannotOpenFile', 'Cannot open file: %s', filename);
end

numVertices = size(vertices, 2);
numFaces = size(faces, 2);
hasNormals = ~isempty(normals);
hasColors = ~isempty(colors);

% Write header
fprintf(fid, '# Exported from StaticFusion\n');
fprintf(fid, '# %d vertices, %d faces\n', numVertices, numFaces);

% Write vertices
for i = 1:numVertices
    fprintf(fid, 'v %.6f %.6f %.6f', vertices(1, i), vertices(2, i), vertices(3, i));
    if hasColors
        fprintf(fid, ' %.3f %.3f %.3f', colors(1, i)/255.0, colors(2, i)/255.0, colors(3, i)/255.0);
    end
    fprintf(fid, '\n');
end

% Write normals (if available)
if hasNormals
    for i = 1:numVertices
        fprintf(fid, 'vn %.6f %.6f %.6f\n', normals(1, i), normals(2, i), normals(3, i));
    end
end

% Write faces
for i = 1:numFaces
    if hasNormals
        fprintf(fid, 'f %d//%d %d//%d %d//%d\n', ...
            faces(1, i), faces(1, i), ...
            faces(2, i), faces(2, i), ...
            faces(3, i), faces(3, i));
    else
        fprintf(fid, 'f %d %d %d\n', faces(1, i), faces(2, i), faces(3, i));
    end
end

fclose(fid);

end

