file1 = "C:\Users\maaya\Desktop\University\labPrut\data\h5\dataControlDay1Try2.h5";
file2 = "C:\Users\maaya\Desktop\University\labPrut\data\h5\dataHfsDay1Try2.h5";

% Read the 'tracks' dataset
combined_tracks = h5read(file1, '/tracks');
combined_tracks2 = h5read(file2, '/tracks');

% Extract the tip coordinates
TIPS1 = squeeze(combined_tracks(:, 7, :));
TIPS2 = squeeze(combined_tracks(:, 8, :));
TIPS3 = squeeze(combined_tracks(:, 9, :));
TIPS4 = squeeze(combined_tracks(:, 10, :));
TIPS5 = squeeze(combined_tracks(:, 11, :));

MCP1 = squeeze(combined_tracks(:, 2, :));
MCP2 = squeeze(combined_tracks(:, 3, :));
MCP3 = squeeze(combined_tracks(:, 4, :));
MCP4 = squeeze(combined_tracks(:, 5, :));
MCP5 = squeeze(combined_tracks(:, 6, :));

TIPS1HFS = squeeze(combined_tracks2(:, 7, :));
TIPS2HFS = squeeze(combined_tracks2(:, 8, :));
TIPS3HFS = squeeze(combined_tracks2(:, 9, :));
TIPS4HFS = squeeze(combined_tracks2(:, 10, :));
TIPS5HFS = squeeze(combined_tracks2(:, 11, :));

MCP1HFS = squeeze(combined_tracks2(:, 2, :));
MCP2HFS = squeeze(combined_tracks2(:, 3, :));
MCP3HFS = squeeze(combined_tracks2(:, 4, :));
MCP4HFS = squeeze(combined_tracks2(:, 5, :));
MCP5HFS = squeeze(combined_tracks2(:, 6, :));

% Calculate the distances between the tip pairs
distances_control = [
    sqrt(sum((TIPS1 - TIPS2).^2, 2)), ...
    sqrt(sum((TIPS1 - TIPS3).^2, 2)), ...
    sqrt(sum((TIPS1 - TIPS4).^2, 2)), ...
    sqrt(sum((TIPS1 - TIPS5).^2, 2)), ...
    sqrt(sum((TIPS2 - TIPS3).^2, 2)), ...
    sqrt(sum((TIPS2 - TIPS4).^2, 2)), ...
    sqrt(sum((TIPS2 - TIPS5).^2, 2)), ...
    sqrt(sum((TIPS3 - TIPS4).^2, 2)), ...
    sqrt(sum((TIPS3 - TIPS5).^2, 2)), ...
    sqrt(sum((TIPS4 - TIPS5).^2, 2))
];

distances_HFS = [
    sqrt(sum((TIPS1HFS - TIPS2HFS).^2, 2)), ...
    sqrt(sum((TIPS1HFS - TIPS3HFS).^2, 2)), ...
    sqrt(sum((TIPS1HFS - TIPS4HFS).^2, 2)), ...
    sqrt(sum((TIPS1HFS - TIPS5HFS).^2, 2)), ...
    sqrt(sum((TIPS2HFS - TIPS3HFS).^2, 2)), ...
    sqrt(sum((TIPS2HFS - TIPS4HFS).^2, 2)), ...
    sqrt(sum((TIPS2HFS - TIPS5HFS).^2, 2)), ...
    sqrt(sum((TIPS3HFS - TIPS4HFS).^2, 2)), ...
    sqrt(sum((TIPS3HFS - TIPS5HFS).^2, 2)), ...
    sqrt(sum((TIPS4HFS - TIPS5HFS).^2, 2))
];

distances_control_tips_mcp = [
    sqrt(sum((TIPS1 - MCP1).^2, 2)), ...
    sqrt(sum((TIPS2 - MCP2).^2, 2)), ...
    sqrt(sum((TIPS3 - MCP3).^2, 2)), ...
    sqrt(sum((TIPS4 - MCP4).^2, 2)), ...
    sqrt(sum((TIPS5 - MCP5).^2, 2))
];

distances_HFS_tips_mcp = [
    sqrt(sum((TIPS1HFS - MCP1HFS).^2, 2)), ...
    sqrt(sum((TIPS2HFS - MCP2HFS).^2, 2)), ...
    sqrt(sum((TIPS3HFS - MCP3HFS).^2, 2)), ...
    sqrt(sum((TIPS4HFS - MCP4HFS).^2, 2)), ...
    sqrt(sum((TIPS5HFS - MCP5HFS).^2, 2))
];

% Compute means for each distance category, skipping NaN values
means_control = mean(distances_control, 'omitnan');
means_HFS = mean(distances_HFS, 'omitnan');

means_control_tips_mcp = mean(distances_control_tips_mcp, 'omitnan');
means_HFS_tips_mcp = mean(distances_HFS_tips_mcp, 'omitnan');

% Combine means into matrices for t-test
means_control_all = [means_control, means_control_tips_mcp];
means_HFS_all = [means_HFS, means_HFS_tips_mcp];

% Find the maximum value for the y-axis limit in the plots
maxValue = max([means_control_all(:); means_HFS_all(:)]);

% Calculate standard deviations for each distance category
std_control = std(distances_control, 'omitnan');
std_HFS = std(distances_HFS, 'omitnan');

std_control_tips_mcp = std(distances_control_tips_mcp, 'omitnan');
std_HFS_tips_mcp = std(distances_HFS_tips_mcp, 'omitnan');

% Plotting figures with increased font sizes
figure;

% First figure
bar([means_control; means_HFS]');
hold on;
x = 1:numel(means_control);
% Add only upper error bars
errorbar(x - 0.15, means_control, zeros(size(std_control)), std_control, 'k.', 'LineWidth', 3);
errorbar(x + 0.15, means_HFS, zeros(size(std_HFS)), std_HFS, 'k.', 'LineWidth', 3);
hold off;
title('Means of Distances Between Tips - Control vs. HFS', 'FontSize', 20); % Adjusted font size
xlabel('TIPS Pairs', 'FontSize', 20); % Adjusted font size
ylabel('Mean Distance', 'FontSize', 20); % Adjusted font size
legend('Control', 'HFS', 'Error Bars', 'FontSize', 20); % Adjusted font size
xticks(1:10);
xticklabels({'1-2', '1-3', '1-4', '1-5', '2-3', '2-4', '2-5', '3-4', '3-5', '4-5'});
set(gca, 'FontSize', 20); % Adjusted font size for tick labels
ylim([0, maxValue + max([std_control(:); std_HFS(:)])]);

% Second figure
figure;
bar([means_control_tips_mcp; means_HFS_tips_mcp]');
hold on;
x = 1:numel(means_control_tips_mcp);
% Add only upper error bars
errorbar(x - 0.15, means_control_tips_mcp, zeros(size(std_control_tips_mcp)), std_control_tips_mcp, 'k.', 'LineWidth', 3);
errorbar(x + 0.15, means_HFS_tips_mcp, zeros(size(std_HFS_tips_mcp)), std_HFS_tips_mcp, 'k.', 'LineWidth', 3);
hold off;
title('Means of Distances Between TIPS and MCP - Control vs. HFS', 'FontSize', 20); % Adjusted font size
xlabel('TIPS - MCP Pairs', 'FontSize', 20); % Adjusted font size
ylabel('Mean Distance', 'FontSize', 20); % Adjusted font size
legend('Control', 'HFS', 'Error Bars', 'FontSize', 20); % Adjusted font size
xticks(1:5);
xticklabels({'Tips1-MCP1', 'Tips2-MCP2', 'Tips3-MCP3', 'Tips4-MCP4', 'Tips5-MCP5'});
set(gca, 'FontSize', 20); % Adjusted font size for tick labels
ylim([0, maxValue + max([std_control_tips_mcp(:); std_HFS_tips_mcp(:)])]);
