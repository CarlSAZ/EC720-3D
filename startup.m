currentFile = mfilename('fullpath');
if isempty(currentFile)
    % When run from the Command Window or editor, fall back to active file.
    try
        currentFile = matlab.desktop.editor.getActiveFilename;
    catch
        currentFile = '';
    end
end
if isempty(currentFile)
    rootDir = pwd;
else
    rootDir = fileparts(currentFile);
end

addpath(fullfile(rootDir, '2DModels'));
addpath(fullfile(rootDir, 'sun3d'));
addpath(fullfile(rootDir, '3D_Utilities'));
addpath(fullfile(rootDir, 'utils'));
addpath(fullfile(rootDir, 'fusion'));
addpath(fullfile(rootDir, 'segmentation'));
addpath(fullfile(rootDir, 'tracking'));
addpath(fullfile(rootDir, 'visualization'));
addpath(fullfile(rootDir, 'reconstruction'));
addpath(fullfile(rootDir, 'BronnMatlab'));