% Source: SUN3D Database
% https://sun3d.cs.princeton.edu/toolbox/
function data = loadSUN3D(sequenceName, frameIDs)

    if ~exist('sequenceName','var')
        sequenceName = 'hotel_umd/maryland_hotel3';
    end

    SUN3Dpath = 'https://sun3d.cs.princeton.edu/data/';

    data.K = reshape(readValuesFromTxt(fullfile(SUN3Dpath,sequenceName,'intrinsics.txt')),3,3)';
    
    imageFiles = dirSmart(fullfile(SUN3Dpath,sequenceName,'image/'),'jpg');
    depthFiles = dirSmart(fullfile(SUN3Dpath,sequenceName,'depth/'),'png');
    extrinsicsFiles = dirSmart(fullfile(SUN3Dpath,sequenceName,'extrinsics/'),'txt');

    imageFrameID = zeros(1,length(imageFiles));
    imageTimestamp = zeros(1,length(imageFiles));
    for i=1:length(imageFiles)
        id_time = sscanf(imageFiles(i).name, '%d-%d.jpg');
        imageFrameID(i) = id_time(1);
        imageTimestamp(i) = id_time(2);
    end
    depthFrameID = zeros(1,length(depthFiles));
    depthTimestamp = zeros(1,length(depthFiles));
    for i=1:length(depthFiles)
        id_time = sscanf(depthFiles(i).name, '%d-%d.png');
        depthFrameID(i) = id_time(1);
        depthTimestamp(i) = id_time(2);
    end
    
    frameCount = length(imageFiles);
    IDimage2depth = zeros(1,frameCount);
    for i=1:frameCount
        [~, IDimage2depth(i)]=min(abs(double(depthTimestamp)-double(imageTimestamp(i))));
    end
    
    data.annotation = annotationRead(fullfile(SUN3Dpath,sequenceName,'annotation/index.json'));
    
    if ~exist('frameIDs','var') || isempty(frameIDs)
        frameIDs = 1:frameCount;
    end
    
    cnt = 0;
    for frameID=frameIDs
        cnt = cnt + 1;
        data.image{cnt} = fullfile(fullfile(SUN3Dpath,sequenceName,'image',imageFiles(frameID).name));
        data.depth{cnt} = fullfile(fullfile(SUN3Dpath,sequenceName,'depth',depthFiles(IDimage2depth(frameID)).name));
    end
    
    try
        data.extrinsicsC2W = permute(reshape(readValuesFromTxt(fullfile(SUN3Dpath,sequenceName,'extrinsics',extrinsicsFiles(end).name)),4,3,[]),[2 1 3]);
    catch
        data.extrinsicsC2W = [];
    end
    
end

function values = readValuesFromTxt(filename)
    try
        values = textscan(webread(filename),'%f');
    catch
        fid = fopen(filename,'r');
        values = textscan(fid,'%f');
        fclose(fid);
    end
    values = values{1};
end


function files = dirSmart(page, tag)
    [files, status] = urldir(page, tag);
    if status == 0
        files = dir(fullfile(page, ['*.' tag]));
    end
end

function [files, status] = urldir(page, tag)
    if nargin == 1
        tag = '/';
    else
        tag = lower(tag);
        if strcmp(tag, 'dir')
            tag = '/';
        end
        if strcmp(tag, 'img')
            tag = 'jpg';
        end
    end
    nl = length(tag);
    nfiles = 0;
    files = [];

    page = strrep(page, '\', '/');
    [webpage, status] = urlread(page);

    if status
        j1 = findstr(lower(webpage), '<a href="');
        j2 = findstr(lower(webpage), '</a>');
        Nelements = length(j1);
        if Nelements>0
            for f = 1:Nelements
                chain = webpage(j1(f):j2(f));
                jc = findstr(lower(chain), '">');
                chain = deblank(chain(10:jc(1)-1));

                if length(chain)>length(tag)-1
                    if strcmp(chain(end-nl+1:end), tag)
                        nfiles = nfiles+1;
                        chain = strrep(chain, '%20', ' ');
                        files(nfiles).name = chain;
                        files(nfiles).bytes = 1;
                    end
                end
            end
        end
    end
end


function annotation = annotationRead(filename)
    annotation = [];
    try
        str = urlread(filename);
        annotation = loadjson(str);
    catch
        try
            str = file2string(filename);
            annotation = loadjson(str);
        catch
        end
    end
end


function fileStr = file2string(fname)
    fileStr = '';
    fid = fopen(fname,'r');
    tline = fgetl(fid);
    while ischar(tline)
        fileStr = [fileStr ' ' tline];
        tline = fgetl(fid);
    end
    fclose(fid);
end


