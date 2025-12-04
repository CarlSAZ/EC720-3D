clear all;close all;
%% this code demonstrates how to load SUN3D and how to use SUN3D in different ways

% data = loadSUN3D('harvard_robotics_lab/hv_s1_1',1:20);
data = loadSUN3D('harvard_tea_1/hv_tea1_2',400+(1:5));
%% Get Semi-truth displacements

focalLen = data.K(1,1);

[x,y] = meshgrid((1:640)-data.K(1,3), (1:480) - data.K(2,3));

img1 = imread(data.image{1});
img2 = imread(data.image{3});

img1Gry = double(im2gray(img1));
img2Gry = double(im2gray(img2));
depth1 = depthRead(data.depth{1});
depth2 = depthRead(data.depth{3});

[RotMat,Tvec] = getRelativeTransform(data.extrinsicsC2W(:,:,1),data.extrinsicsC2W(:,:,4));

[delta,xp,yp] = affineMotion(RotMat,Tvec,focalLen,x,y,depth1);

figure(1);clf;
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(delta(1,1:10:end,1:10:end)),-squeeze(delta(2,1:10:end,1:10:end)),'off','k');
colormap gray
title("Ground Truth optical flow",{sprintf("\\theta_z = %.3f, \\theta_y = %.3f, \\theta_x = %.3f",...
    eul(1),eul(2),eul(3));
    sprintf("T_x = %.3f, T_y = %.3f, T_z = %.3f",Tvec(1),Tvec(2),Tvec(3))})
%% Try Grayscale Horne-Schunk

[u,v,cost,mse,mae,Uhist,Vhist] = HornSchunk(img1Gry,img2Gry,20);

figure(2);clf;
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-u(1:10:end,1:10:end),-v(1:10:end,1:10:end),'off','k');
colormap gray;
%% Try Block Displacement

[Dabs, dBlockOpt] = BlockSearch(img1Gry,img2Gry,[20,20],16,@abs);

[offsetWhole] = crossCorrEst(img1Gry,img2Gry);
deltaCorrWhole = repmat(offsetWhole(:,1),[1 size(img1Gry)]);
[deltaCorr,blockDiffs,phaseDiffAll] = subCrossCorrEst(img1Gry,img2Gry,[20 20]);

figure(3);clf;
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(Dabs(1,1:10:end,1:10:end)),-squeeze(Dabs(2,1:10:end,1:10:end)),'off','k');
colormap gray;


%% 

eul = rotm2eul(RotMat);

figure(10);
subplot(3,2,1)
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(delta(1,1:10:end,1:10:end)),-squeeze(delta(2,1:10:end,1:10:end)),'off','k');
colormap gray
title("Ground Truth optical flow",{sprintf("\\theta_z = %.3f, \\theta_y = %.3f, \\theta_x = %.3f",...
    eul(1),eul(2),eul(3));
    sprintf("T_x = %.3f, T_y = %.3f, T_z = %.3f",Tvec(1),Tvec(2),Tvec(3))})

subplot(3,2,2)
imagesc(x(:,1),y(1,:),depth1);
clim([1 2.2])
colorbar;
title("Depth")


subplot(3,2,3)
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-u(1:10:end,1:10:end),-v(1:10:end,1:10:end),'off','k');
colormap gray;
title("Horne-Schunk estimation")


subplot(3,2,4)
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(Dabs(1,1:10:end,1:10:end)),-squeeze(Dabs(2,1:10:end,1:10:end)),'off','k');
colormap gray;
title("Exhaustive Block Displacement")

subplot(3,2,5)
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(deltaCorrWhole(1,1:10:end,1:10:end)),-squeeze(deltaCorrWhole(2,1:10:end,1:10:end)),'off','k');
colormap gray;
title("Cross correlation - whole")

subplot(3,2,6)
imagesc(x(:,1),y(1,:),img1Gry);
hold on;
quiver(x(1:10:end,1:10:end),y(1:10:end,1:10:end),-squeeze(deltaCorr(1,1:10:end,1:10:end)),-squeeze(deltaCorr(2,1:10:end,1:10:end)),'off','k');
colormap gray;
title("Cross correlation - 20x20 blocks")


