clear all;close all;
groundTruthFile = "D:\Bronn\rgbd_bonn_groundtruth_1mm_section.ply";
dataset = "D:\Bronn\rgbd_bonn_crowd";

mkdir(fullfile(dataset,'motionLabels'));
mkdir(fullfile(dataset,'distFromModel'));
load("BronnTruth.mat");
%%

fid = fopen(fullfile(dataset,'rgb.txt'));
temp = textscan(fid,'%f %s','HeaderLines',2);
rgbFrameList = table(temp{1},string(temp{2}),'VariableNames',{'time_posix','filename'});
fclose(fid);
fid = fopen(fullfile(dataset,'depth.txt'));
temp = textscan(fid,'%f %s','HeaderLines',2);
depthFrameList = table(temp{1},string(temp{2}),'VariableNames',{'time_posix','filename'});
fclose(fid);
%% Pose truth
fid = fopen(fullfile(dataset,"groundtruth.txt"));
% timestamp tx ty tz qx qy qz qw
temp = textscan(fid,"%f %f %f %f %f %f %f %f",'HeaderLines',2);
fclose(fid);
poseTruth = table(temp{1},[temp{2:4}],[temp{8},temp{5:7}],'VariableNames',{'timestamp','txyz','quat'});
%%
fx = 542.822841;
fy = 542.576870;
cx = 315.593520;
cy = 237.756098;

for rgbIdx = 1:height(rgbFrameList)
   t1 = rgbFrameList.time_posix(rgbIdx);
[~,depthIdx] = min(abs(t1 - depthFrameList.time_posix));
[~,poseIdx] = min(abs(t1 - poseTruth.timestamp));

im = imread(fullfile(dataset,rgbFrameList.filename(rgbIdx)));
depth = depthReadTUM(fullfile(dataset,depthFrameList.filename(depthIdx)));

XYZcamera = depth2XYZcameraTUM(fx,fy,cx,cy, depth);
tic;
[labels,dist] = findNonStaticTree(XYZcamera,poseTruth(poseIdx,:),TruthTree,[0.1 0.3]);
toc;

[~,labelname,~] = fileparts(rgbFrameList.filename(rgbIdx));
imwrite(uint8(labels),fullfile(dataset,'motionLabels',strcat(labelname,".png")));
imwrite(dist,fullfile(dataset,'distFromModel',strcat(labelname,".png")));
end