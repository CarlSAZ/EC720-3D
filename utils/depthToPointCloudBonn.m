function [XYZ, RGB] = depthToPointCloudBonn(image, depth, varargin)
%DEPTHTOPOINTCLOUD Convert RGB-D data into a 3xN point cloud with colours.
%   [XYZ, RGB] = DEPTHTOPOINTCLOUD(image, depth, K, ...) applies depth
%   filtering, back-projects valid pixels to 3D camera coordinates, and
%   returns XYZ in metres and RGB in the range [0,1].
%
%   Name-value pairs:
%       'depthFilter'   - struct passed to filterDepth
%       'keepMask'      - logical mask (same size as depth) to pre-filter

narginchk(3, inf);

validateattributes(image, {'uint8'}, {'size',[NaN NaN 3]}, mfilename, 'image', 1);
validateattributes(depth, {'double','single'}, {'2d'}, mfilename, 'depth', 2);

if ~isa(depth, 'double')
    depth = double(depth);
end

p = inputParser;
addParameter(p, 'depthFilter', struct());
addParameter(p, 'keepMask', []);
parse(p, varargin{:});
opts = p.Results;

% Magic parameters for bonn dataset
fx = 542.822841;
fy = 542.576870;
cx = 315.593520;
cy = 237.756098;

depthFilterArgs = structToNameValue(opts.depthFilter);
depthFiltered = filterDepth(depth, depthFilterArgs{:});


XYZcamera = depth2XYZcameraTUM(fx,fy,cx,cy, depthFiltered);
XYZcamera = reshape(XYZcamera,[],4);

valid = (XYZcamera(:,3) > 0);
if ~isempty(opts.keepMask)
    valid = valid & opts.keepMask;
end

XYZ = XYZcamera(valid,1:3).';

imgReshaped = double(reshape(image, [], 3));
RGB = imgReshaped(valid(:), :)' / 255;

end

function args = structToNameValue(s)
if isempty(s)
    args = {};
    return;
end
if ~isstruct(s) || numel(s) ~= 1
    error('Expected a scalar struct.');
end
fields = fieldnames(s);
args = cell(1, numel(fields) * 2);
for idx = 1:numel(fields)
    args{2*idx - 1} = fields{idx};
    args{2*idx} = s.(fields{idx});
end
end

