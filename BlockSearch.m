function [delta,D] = BlockSearch(frame1,frame2,blockSize,maxDist,errFunc)
    [X,Y] = meshgrid(-maxDist:maxDist,-maxDist:maxDist);
    X = X(:);
    Y = Y(:);
    
    % Pad the moving frame
    frame1 = padarray(frame1,[maxDist,maxDist],0,'both');

    nX = size(frame2,1);
    nY = size(frame2,2);

    numBlocks = ceil(size(frame2)./blockSize);
    blockDiffs = zeros([numBlocks,numel(X)]);
    for dIdx = 1:numel(X)
        diffFrame = frame2 - frame1(maxDist+1-X(dIdx):maxDist+1+nX-X(dIdx)-1,maxDist+1-Y(dIdx):maxDist+1+nY-Y(dIdx)-1);
        diffFrame = errFunc(diffFrame);
        for ii = 1:numBlocks(1)
            for jj = 1:numBlocks(2)
                xIdx1 = (ii-1)*blockSize(1)+1;
                xIdx2 = xIdx1 + blockSize(1)-1;
                yIdx1 = (jj-1)*blockSize(2)+1;
                yIdx2 = yIdx1 + blockSize(2)-1;
                blockDiffs(ii,jj,dIdx) = sum(diffFrame(xIdx1:xIdx2,yIdx1:yIdx2),'all');

            end
        end
    end

    [~,bestIdx] = min(blockDiffs,[],3);
    D(1,:,:) = X(bestIdx);
    D(2,:,:) = Y(bestIdx);

    delta = zeros([2,size(frame2)]);
     for ii = 1:numBlocks(1)
            for jj = 1:numBlocks(2)
                xIdx1 = (ii-1)*blockSize(1)+1;
                xIdx2 = xIdx1 + blockSize(1)-1;
                yIdx1 = (jj-1)*blockSize(2)+1;
                yIdx2 = yIdx1 + blockSize(2)-1;

                delta(:,xIdx1:xIdx2,yIdx1:yIdx2) = repmat(D(:,ii,jj),[1,blockSize]);
            end
     end
end