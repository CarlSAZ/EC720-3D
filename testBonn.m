

groundTruthFile = "C:\Users\carl\Documents\EC720\rgbd_bonn_dataset\rgbd_bonn_groundtruth_1mm_section.ply";
dataset = "C:\Users\carl\Documents\EC720\rgbd_bonn_dataset\rgbd_bonn_crowd";

% groundTruthFile = "D:\Bronn\rgbd_bonn_groundtruth_1mm_section.ply";
% dataset = "D:\Bronn\rgbd_bonn_crowd";

if exist("BronnTruth.mat",'file')
    load("BronnTruth.mat");
else
fid = fopen(groundTruthFile);
data = textscan(fid,"%f %f %f %u8 %u8 %u8 %f32",'HeaderLines',13);
fclose(fid);

truthTable = table([data{1},data{2},data{3}],[data{4},data{5},data{6}],data{7},...
    'VariableNames',{'xyz','rgb','scalar'});
end
TruthTree = KDTreeSearcher(truthTable.xyz);
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
t1 = 1548339829.86292;
[~,rgbIdx] = min(abs(t1 - rgbFrameList.time_posix));
[~,depthIdx] = min(abs(t1 - depthFrameList.time_posix));
%%
% rgbIdx = 1;
% depthIdx = 2;
offset = rgbFrameList.time_posix(rgbIdx) - depthFrameList.time_posix(depthIdx);

im = imread(fullfile(dataset,rgbFrameList.filename(rgbIdx)));
depth = depthReadTUM(fullfile(dataset,depthFrameList.filename(depthIdx)));

fx = 542.822841;
fy = 542.576870;
cx = 315.593520;
cy = 237.756098;
XYZcamera = depth2XYZcameraTUM(fx,fy,cx,cy, depth);

[~,poseIdx] = min(abs(t1 - poseTruth.timestamp));

[labels,dist] = findNonStaticTree(XYZcamera,poseTruth(poseIdx,:),TruthTree,[0.1 0.3]);

Tg = bronnTransform(poseTruth(poseIdx,:));
newTruth = truthTable;
newTruth.xyz = [truthTable.xyz ones(height(truthTable),1)] * inv(Tg.');
%%
figure(1);clf;
scatter3(reshape(XYZcamera(:,:,1),[],1), ...
    reshape(XYZcamera(:,:,2),[],1),...
    reshape(XYZcamera(:,:,3),[],1),[], ...
    reshape(double(im)/255,[],3),'.');
hold on;
scatter3(newTruth.xyz(1:10:end,1),newTruth.xyz(1:10:end,2),newTruth.xyz(1:10:end,3),[],double(newTruth.rgb(1:10:end,:))./255,'.')