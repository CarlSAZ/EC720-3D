function [labels,dist] = findNonStatic(CameraXYZ,pose,truthTable,thresh)

Tg = bronnTransform(pose);

origDim = size(CameraXYZ);

CameraXYZ = reshape(CameraXYZ,[],4) * Tg.';

if 0

figure(1);clf;
scatter3(CameraXYZ(:,1), ...
    CameraXYZ(:,2),...
    CameraXYZ(:,3),'k.'); %reshape(double(im)/255,[],3)
hold on;
scatter3(truthTable.xyz(1:20:end,1),truthTable.xyz(1:20:end,2),truthTable.xyz(1:20:end,3), ...
    [],double(truthTable.rgb(1:20:end,:))./255,'.')
end

labels = zeros(origDim(1:2));
dist = zeros(origDim(1:2));
for idx = 1:size(CameraXYZ,1)
    if all(CameraXYZ(idx,1:3) ==0)
        labels(idx) = -1;
        dist(idx) = -1;
        continue
    end

    [mindist,minIdx] = min(vecnorm(CameraXYZ(idx,1:3) - truthTable.xyz,1,2));
    dist(idx) = mindist;
    if mindist > thresh
        labels(idx) = 1;
    end
end