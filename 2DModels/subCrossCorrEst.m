% Author: Carl Stevenson (HW problem solution)
function [delta,blockDiffs,dBlockAll] = subCrossCorrEst(origIm,nextIm,blockSize)
% 
[f1,f2] = freqspace(64,'meshgrid');
Hd = ones(64);
Hd(abs(f1) > 0.4 | abs(f2) > 0.4) =0;
win = fspecial('gaussian',64,2);
win = win./ max(win(:));
H = fwind2(Hd,win);

origIm = filter2(H,origIm);
nextIm = filter2(H,nextIm);

numBlocks = ceil(size(origIm)./blockSize);

delta = zeros([2,size(origIm)]);
blockDiffs = zeros([2,numBlocks]);
dBlockAll = zeros([2,numBlocks,3]);
for ii = 1:numBlocks(1)
    for jj = 1:numBlocks(2)
        xIdx1 = (ii-1)*blockSize(1)+1;
        xIdx2 = xIdx1 + blockSize(1)-1;
        yIdx1 = (jj-1)*blockSize(2)+1;
        yIdx2 = yIdx1 + blockSize(2)-1;
        frame1 = origIm(xIdx1:xIdx2,yIdx1:yIdx2);
        frame2 = nextIm(xIdx1:xIdx2,yIdx1:yIdx2);

        [offset,Ruu] = crossCorrEst(frame1,frame2);
        mae = zeros(1,size(offset,2));
        for idx = 1:size(offset,2)
            
            % do this the dumb way beacuse its easy to code
            pred = zeros(size(frame1));
            for subII = 1:blockSize(1)
                if xIdx1+subII-1 -offset(1,idx) < 1 || xIdx1+subII-1 -offset(1,idx) > size(origIm,1)
                    continue;
                end
                for subJJ = max(1,offset(2,idx)+1):min(blockSize(2),blockSize(2)+offset(2,idx))

                    if yIdx1+subJJ-1 -offset(2,idx) < 1 || yIdx1+subJJ-1 -offset(2,idx) > size(origIm,2)
                        continue;
                    end
                    pred(subII,subJJ) =origIm(xIdx1+subII-1-offset(1,idx),yIdx1+subJJ-1-offset(2,idx));
                end
            end
            valid = pred > 0;
            mae(idx) = sum(abs(frame2(valid) - pred(valid)),'all');
        end
        [~,bestOffset] = min(mae);
        delta(:,xIdx1:xIdx2,yIdx1:yIdx2) = repmat(offset(:,bestOffset),[1,blockSize]);
        blockDiffs(:,ii,jj) = offset(:,bestOffset);
        dBlockAll(:,ii,jj,:) = permute(offset,[1 3 4 2]);
    end
end
end