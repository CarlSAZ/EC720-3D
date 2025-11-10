function shiftedIm = shiftImageHorne(frame,u,v)
    shiftedIm = zeros(size(frame));

    % Pad the moving frame
    frame = padarray(frame,[10,10],0,'both');

    for ii = 1:size(shiftedIm,1)
        for jj = 1:size(shiftedIm,2)
            shiftedIm(ii,jj) = frame(10+ii-v(ii,jj),10+jj-u(ii,jj));
        end
    end
end