% Generates less dense datapoints so that data is evenly dispersed

%% Read in table
origTable = readtable('LaundryTable2.csv');  % Feed in raw data (dense)
origTable = table2array(origTable);         % Convert to array
newTable = origTable;                       % Copy table for modification

xCoord = origTable(:,8);
yCoord = origTable(:,9);
zCoord = origTable(:,10);
%% Delete points that are too close

len = length(origTable);
border = .01;
count = 1;
toDelete = [];
% xBorder = .002;
% yBorder = .002;
% zBorder = .001;

while count < len - 1
    len = length(newTable);
    deleteCount = count + 1;                % Set next inquiry point
    while deleteCount < len                 % Calculate distance of other point
        distance = sqrt((newTable(count,8) - newTable(deleteCount,8))^2 + (newTable(count,9) - newTable(deleteCount,9))^2 + ...
            (newTable(count,10) - newTable(deleteCount,10))^2);
        %xDist = abs(newTable(count,8) - newTable(deleteCount,8));
        %yDist = abs(newTable(count,9) - newTable(deleteCount,9));
        %zDist = abs(newTable(count,10) - newTable(deleteCount,10));
        if distance < border                % Add index to delete matrix if it's too close
        %if xDist < xBorder && yDist < yBorder && zDist < zBorder
            toDelete = [toDelete, deleteCount];
        end     
        deleteCount = deleteCount + 1;      % Iterate through all other points
    end
    
    for i = length(toDelete):-1:1           % Delete those rows of the table
        newTable(toDelete(i),:) = [];
    end
    
    toDelete = [];                          % Reset delete matrix
    count = count + 1;
end

%% Plots

% Original
xCoord = origTable(:,8);
yCoord = origTable(:,9);
zCoord = origTable(:,10);

figure()
plot3(xCoord, yCoord, zCoord, '.')
grid on;
xlabel('x (m)', 'Interpreter','latex')
ylabel('y (m)', 'Interpreter','latex')
zlabel('z (m)', 'Interpreter','latex')
title('Preprocessed Dataset: End Effector Position', 'Interpreter','latex')

% Modified
xNew = newTable(:,8);
yNew = newTable(:,9);
zNew = newTable(:,10);

figure()
plot3(xNew, yNew, zNew, 'r.')
grid on;
xlabel('x (m)', 'Interpreter','latex')
ylabel('y (m)', 'Interpreter','latex')
zlabel('z (m)', 'Interpreter','latex')
title('Processed Dataset: End Effector Position', 'Interpreter','latex')

%% Export
writematrix(newTable,'LaundryTable2Processed_LowerDensity.csv') 