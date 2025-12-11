%% Test script to verify bonnTransform coordinate system
% This script directly tests the coordinate transformation without
% referencing other potentially incorrect files

clear; close all; clc;

% Load Bonn data
scriptDir = fileparts(mfilename('fullpath'));
sequencePath = fullfile(scriptDir, 'rgbd_bonn_crowd');
data = loadBonn(sequencePath, 1:5);  % Load first 5 frames

% Get first frame pose
poseIdx = 1;
poseRow = data.poseTruth(poseIdx, :);

fprintf('=== Testing bonnTransform ===\n');
fprintf('Ground truth pose:\n');
fprintf('  Translation: [%.4f, %.4f, %.4f]\n', poseRow.txyz);
fprintf('  Quaternion: [%.4f, %.4f, %.4f, %.4f]\n', poseRow.quat);

% Get transformation matrix
Tg = bronnTransform(poseRow);
fprintf('\nTransformation matrix Tg:\n');
disp(Tg);

% Check if it's a valid rotation matrix
R = Tg(1:3, 1:3);
fprintf('Rotation matrix determinant: %.6f (should be 1.0)\n', det(R));
fprintf('Rotation matrix is orthogonal: %d (should be 1)\n', ...
    all(abs(R * R' - eye(3)) < 1e-6, 'all'));

% Test point transformation
% Load first frame depth
image = imread(data.image{1});
depth = depthReadTUM(data.depth{1});

% Convert to point cloud (camera coordinates, 3xN format)
fx = 542.822841;
fy = 542.576870;
cx = 315.593520;
cy = 237.756098;
XYZcamera = depth2XYZcameraTUM(fx, fy, cx, cy, depth);
valid = XYZcamera(:,:,3) > 0;
XYZcam = [XYZcamera(valid)'; XYZcamera(valid,2)'; XYZcamera(valid,3)'];
XYZcam = XYZcam(1:3, 1:min(1000, size(XYZcam,2)));  % Sample points

fprintf('\n=== Testing coordinate transformation ===\n');
fprintf('Camera point cloud: %d points\n', size(XYZcam, 2));
fprintf('Camera point range: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]\n', ...
    min(XYZcam(1,:)), max(XYZcam(1,:)), ...
    min(XYZcam(2,:)), max(XYZcam(2,:)), ...
    min(XYZcam(3,:)), max(XYZcam(3,:)));

% Test 1: Column vector format (what StaticFusion uses)
% X_world = T * X_camera
XYZworld1 = Tg(1:3, 1:3) * XYZcam + Tg(1:3, 4);
fprintf('\nMethod 1 (Column vector, T * X):\n');
fprintf('World point range: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]\n', ...
    min(XYZworld1(1,:)), max(XYZworld1(1,:)), ...
    min(XYZworld1(2,:)), max(XYZworld1(2,:)), ...
    min(XYZworld1(3,:)), max(XYZworld1(3,:)));

% Test 2: Row vector format (what testBonn.m uses)
% X_world = X_camera * T'
XYZcam_row = XYZcam';  % Convert to row vectors (Nx3)
XYZcam_row_hom = [XYZcam_row, ones(size(XYZcam_row,1), 1)];  % Nx4 homogeneous
XYZworld2_hom = XYZcam_row_hom * Tg';
XYZworld2 = XYZworld2_hom(:, 1:3)';
fprintf('\nMethod 2 (Row vector, X * T''):\n');
fprintf('World point range: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]\n', ...
    min(XYZworld2(1,:)), max(XYZworld2(1,:)), ...
    min(XYZworld2(2,:)), max(XYZworld2(2,:)), ...
    min(XYZworld2(3,:)), max(XYZworld2(3,:)));

% Test 3: Using transpose of Tg
XYZworld3 = Tg'(1:3, 1:3) * XYZcam + Tg'(1:3, 4);
fprintf('\nMethod 3 (Column vector, T'' * X):\n');
fprintf('World point range: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]\n', ...
    min(XYZworld3(1,:)), max(XYZworld3(1,:)), ...
    min(XYZworld3(2,:)), max(XYZworld3(2,:)), ...
    min(XYZworld3(3,:)), max(XYZworld3(3,:)));

% Check camera position
fprintf('\n=== Camera position in world frame ===\n');
fprintf('Method 1 (T * [0;0;0]): [%.4f, %.4f, %.4f]\n', Tg(1:3, 4));
fprintf('Method 2 (X * T'' where X=[0,0,0,1]): [%.4f, %.4f, %.4f]\n', ...
    [0, 0, 0, 1] * Tg');
fprintf('Method 3 (T'' * [0;0;0]): [%.4f, %.4f, %.4f]\n', Tg'(1:3, 4));
fprintf('Ground truth translation: [%.4f, %.4f, %.4f]\n', poseRow.txyz);

fprintf('\n=== Analysis ===\n');
fprintf('Note: The correct method should produce reasonable world coordinates.\n');
fprintf('Camera position should match ground truth translation (after coordinate transform).\n');
