% Author: Carl Stevenson (HW problem solution)
function [offset,Ruu,x,y] = crossCorrEst(frame1,frame2)

    win1 = window(@hann,size(frame1,1));
    win2 = window(@hann,size(frame1,2));

    mywin = win1*win2';
    mywin = mywin ./max(mywin,[],'all');
    nfftX = 2*size(frame1,1);
    nfftY = 2*size(frame2,2);
    ft1 = fft2(frame1.*mywin,nfftX,nfftY);
    ft2 = fft2(frame2.*mywin,nfftX,nfftY);

    Psi_hat = conj(ft1).*ft2;
    Psi_hat = Psi_hat./abs(Psi_hat);

    % fix 0 issues
    badix = isnan(Psi_hat);
    Psi_hat(badix) = exp(1i*(angle(ft2(badix))-angle(ft1(badix))));

    Ruu = ifft2(Psi_hat,'symmetric');

    % Translate to [-m/2 m/2] by shifting Ruu
    Ruu = ifftshift(ifftshift(Ruu,1),2);
    Ruu = Ruu(nfftX/4 + (1:size(frame1,1)),nfftY/4 + (1:size(frame1,2)));

    maxRuu = movmax(movmax(Ruu,3,2),3,1);

    Map = Ruu >= maxRuu;
    Prom = Ruu;

    % [Map,Prom] = islocalmax2(Ruu,'MaxNumExtrema',3,'MinSeparation',3);
    ind = find(Map);
    [~,tmp] = sort(Ruu(ind),'descend');
    ind = ind(tmp);
    [dr,dc] = ind2sub(size(Ruu),ind);

    x = (0:size(Ruu,1)-1) - floor(size(Ruu,1)/2);
    y = (0:size(Ruu,2)-1) - floor(size(Ruu,2)/2);

    offset(1,:) = x(dr(1:3));
    offset(2,:) = y(dc(1:3));

end