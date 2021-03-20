% First, run transforms.m to get finalTrans matrix
% This code generates EEF position (x,y,z) from input joint angles

%% Read in Joint Positions
jointPositions = readtable('JointPositionLaundry3.csv'); % Feed in raw data
jointPositions = table2array(jointPositions(:,2:8)); % Get rid of non-rotation inputs

%% Calculate EEF position
EEFPosition = zeros(length(jointPositions),3);      % table for EEF positions

q = sym('q', [n 1], 'real'); % generalized coordinates (joint angles)

for i = 1:length(EEFPosition)
    q0 = jointPositions(i,:);           % Input joint coordinates
    position = subs(finalTrans, q, q0');
    Euclidcoord = position * [0;0;0;1]; % Find EEF position
    EEFPosition(i,1) = double(Euclidcoord(1));
    EEFPosition(i,2) = double(Euclidcoord(2));
    EEFPosition(i,3) = double(Euclidcoord(3));
end

%% Export dataset to one file
% First 7 columns are input angles, last 3 columns are EEF position (x,y,z)
totaltable = zeros(length(jointPositions), width(jointPositions) + width(EEFPosition));
totaltable(:,1:width(jointPositions)) = jointPositions;
totaltable(:,width(jointPositions)+1:end) = EEFPosition;
writematrix(totaltable,'LaundryTable3.csv') 