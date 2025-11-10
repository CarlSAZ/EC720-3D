function shiftedIm = shiftImage(frame,Displacement,maxDist)
    shiftedIm = zeros(size(frame));

    % Pad the moving frame
    frame = padarray(frame,[maxDist,maxDist],0,'both');

    for ii = 1:size(shiftedIm,1)
        for jj = 1:size(shiftedIm,2)
            shiftedIm(ii,jj) = frame(maxDist+ii-Displacement(1,ii,jj),maxDist+jj-Displacement(2,ii,jj));
        end
    end
end