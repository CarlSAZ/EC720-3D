% Author: Carl Stevenson
function plotPoses(truth,estimated)

figure(33);clf;

tx = squeeze(truth(1,4,:));
ty = squeeze(truth(2,4,:));
tz = squeeze(truth(3,4,:));
plot3(tx,ty,tz,'-k.')
hold on;
ex = squeeze(estimated(1,4,:));
ey = squeeze(estimated(2,4,:));
ez = squeeze(estimated(3,4,:));
plot3(ex,ey,ez,'-r.')
hold on;

for idx = 1:size(truth,3)
    tmp = squeeze(truth(1:3,1:3,idx))*[0,0,1]';
    tv(:,idx) = tmp;
end
for idx = 1:size(estimated,3)
    tmp = squeeze(estimated(1:3,1:3,idx))*[0,0,1]';
    ev(:,idx) = tmp;
end

quiver3(tx(1:10:end),ty(1:10:end),tz(1:10:end),tv(1,1:10:end).',tv(2,1:10:end).',tv(3,1:10:end).','off','k');
quiver3(ex,ey,ez,ev(1,:).',ev(2,:).',ev(3,:).','off','r');