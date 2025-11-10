function [u,v,cost,mse,mae,Uhist,Vhist] = HornSchunk(image1,image2,lambda)

    u = zeros(size(image2)+2);
    v = zeros(size(image2)+2);

    Uhist = zeros([size(image2),200]);
    Vhist = zeros([size(image2),200]);

    pU = u;
    pV = v;
    It = image2 - image1;
    [Ix,Iy] = gradient(image2);

    cost = zeros(1,100);
    mse = zeros(2,100);
    mae = zeros(2,100);
    for itter = 1:200
        uBar = 0.25*(pU(3:end,2:end-1) + pU(1:end-2,2:end-1) + pU(2:end-1,3:end) + pU(2:end-1,1:end-2));
        vBar = 0.25*(pV(3:end,2:end-1) + pV(1:end-2,2:end-1) + pV(2:end-1,3:end) + pV(2:end-1,1:end-2));

        ratio = (Ix.*uBar + Iy.*vBar + It)./(8*lambda + Ix.^2 + Iy.^2);

        u(2:end-1,2:end-1) = uBar - Ix.*ratio;
        v(2:end-1,2:end-1) = vBar - Iy.*ratio;

        pU = u;
        pV = v;

        error = (Ix.*u(2:end-1,2:end-1) + Iy.*v(2:end-1,2:end-1) + It).^2 + lambda*(...
            ((u(2:end-1,2:end-1) - u(1:end-2,2:end-1)).^2 + (u(2:end-1,2:end-1) - u(2:end-1,1:end-2)).^2 + ...
            (v(2:end-1,2:end-1) - v(1:end-2,2:end-1)).^2 + (v(2:end-1,2:end-1) - v(2:end-1,1:end-2)).^2));
        cost(itter) = sum(error,'all');

        mse(1,itter) = mean((u-0.5).^2 ,'all');
        mse(2,itter) = mean( (v-0.5).^2,'all');
        mae(1,itter) = mean(abs(u-0.5) ,'all');
        mae(2,itter) = mean(abs(v-0.5),'all');

        Uhist(:,:,itter) = u(2:end-1,2:end-1);
        Vhist(:,:,itter) = v(2:end-1,2:end-1);
    end

    u = u(2:end-1,2:end-1);
    v = v(2:end-1,2:end-1);
end