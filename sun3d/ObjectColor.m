function color = ObjectColor(objectID)
objectColors = loadjson(urlread('https://sun3d.cs.princeton.edu/player/ObjectColors.json'));

objectID = mod(objectID,length(objectColors)) + 1;

color = objectColors{objectID};

color = [hex2dec(color(2:3)) hex2dec(color(4:5)) hex2dec(color(6:7))]/255;

end

