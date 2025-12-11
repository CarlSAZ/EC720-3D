function prediction = predictStaticMap(map, Twc, varargin)
% Transform static map surfels into current camera frame

narginchk(2, inf);
validateStaticMap(map);
validateattributes(Twc, {'double'}, {'size',[4,4]}, mfilename, 'Twc', 2);

parser = inputParser;
addParameter(parser, 'ReturnWorld', true, @(x) islogical(x) && isscalar(x));
parse(parser, varargin{:});
opts = parser.Results;

Rcw = inv(Twc(1:3, 1:3));
tcw = -Rcw * Twc(1:3, 4);

prediction = struct();
if isempty(map.positions)
    prediction.XYZcam = zeros(3, 0);
    prediction.normalsCam = zeros(3, 0);
    prediction.colours = zeros(3, 0);
    prediction.confidence = zeros(1, 0);
    prediction.radius = zeros(1, 0);
else
    prediction.XYZcam = Rcw * map.positions + tcw;
    prediction.normalsCam = Rcw * map.normals;
    prediction.colours = map.colours;
    prediction.confidence = map.confidence;
    prediction.radius = map.radius;
end

if opts.ReturnWorld
    prediction.XYZworld = map.positions;
    prediction.normalsWorld = map.normals;
end
end

function validateStaticMap(map)
if ~isstruct(map)
    error('Static map must be a struct.');
end
requiredFields = {'positions', 'normals', 'colours', 'confidence', 'radius', 'params'};
for f = requiredFields
    if ~isfield(map, f{1})
        error('predictStaticMap:MissingField', ...
            'Static map struct missing field ''%s''.', f{1});
    end
end
end

