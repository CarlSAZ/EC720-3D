clear all;close all;
%% this code demonstrates how to load SUN3D and how to use SUN3D in different ways

data = loadSUN3D('harvard_robotics_lab/hv_s1_1',1:20);

%% Get Semi-truth displacements

focalLen = data.K(1,1);

[x,y] = meshgrid((1:640)-data.K(1,3), (1:480) - data.K(2,3));

img1 = imread(data.image{1});
img2 = imread(data.image{4});

img1Gry = double(im2gray(img1));
img2Gry = double(im2gray(img2));
depth1 = depthRead(data.depth{1});
depth2 = depthRead(data.depth{2});

[RotMat,Tvec] = getRelativeTransform(data.extrinsicsC2W(:,:,1),data.extrinsicsC2W(:,:,4));

[delta,xp,yp] = affineMotion(RotMat,Tvec,focalLen,x,y,depth1);

figure(1);clf;
imagesc(x(:,1),y(1,:),img1);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(delta(1,1:10:end,1:10:end)),-squeeze(delta(2,1:10:end,1:10:end)),'off','k');

%% Try Grayscale Horne-Schunk

[u,v,cost,mse,mae,Uhist,Vhist] = HornSchunk(img1Gry,img2Gry,20);

figure(2);clf;
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-u(1:10:end,1:10:end),-v(1:10:end,1:10:end),'off','k');
colormap gray;
%% Try Block Displacement

[Dabs, dBlockOpt] = BlockSearch(img1Gry,img2Gry,[20,20],16,@abs);

figure(3);clf;
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(Dabs(1,1:10:end,1:10:end)),-squeeze(Dabs(2,1:10:end,1:10:end)),'off','k');
colormap gray;
