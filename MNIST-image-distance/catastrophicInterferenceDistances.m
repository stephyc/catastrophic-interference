%% Load MNIST

% original images
ims = loadMNISTImages("train-images-idx3-ubyte");
labels = loadMNISTLabels("train-labels-idx1-ubyte");

%% make variables
nums = cell(8, 10);
fullmean = zeros(28, 28, 8);
distances = cell(8, 10);

%% loop over all manipulations
maniplabels = {'Original', 'Fliplr', 'Flipud', 'Rot90', 'Inverse', 'Checkerboard', 'Invbot', 'Swapud'};

for j = 0:7
   nims = manipMNIST(ims, j);
   
   % Separate by digit
   for i = 0:9
       nums{j + 1, i + 1} = nims(:, :, labels == i);
   end
   
   % Mean of all images
   fullmean = mean(nims, 3);
   
   % fill distances with structs
   for i = 1:10
        distances{j + 1, i} = struct;
   end
    
   for i = 1:10
       % mean image per digit
       tempim = mean(nums{j + 1, i}(:, :, :), 3);
       distances{j + 1, i}.avg = tempim;
    
       % difference from mean (image 1)
       tempim = nums{j + 1, i}(:, :, 1) - distances{j + 1, i}.avg;
       distances{j + 1, i}.diffIm = tempim;
   end
end

%% calculate average ssimval for each image in each dataset
for j = 0:7
    for i = 1:10
        sum = 0;
        for image = 1:size(nums{j + 1, i}, 3)
            t = ssim(nums{j + 1, i}(:, :, image), distances{j + 1, i}.avg);
            sum = sum + t;
        end
        distances{j + 1, i}.ssimval = sum/size(nums{j + 1, i}, 3);
    end
end

%% linearize each image

for i = 1:8
    for j = 1:10
        distances{i, j}.linim = reshape(distances{i, j}.avg, [1, 28 * 28]);
    end
end

%% Correlation within datasets

for i = 1:8
    for j = 1:10
        distances{i, j}.incorrs = zeros(1, 10);
        for k = 1:10
            distances{i, j}.incorrs(k) = corr(distances{i, j}.linim', distances{i, k}.linim');
        end
    end
end

%% Correlation with same number, other datasets

for i = 1:8
    for j = 1:10
        distances{i, j}.outcorrs = zeros(1, 10);
        for k = 1:8
            distances{i, j}.outcorrs(k) = corr(distances{i, j}.linim', distances{k, j}.linim');
        end
    end
end

%% Z-score manip-manip correlations

for manip1 = 1:8
    for digit = 1:10
        distances{manip1, digit}.zoutcorrs = zeros(1, 10);
        for manip2 = 1:8
            distances{manip1, digit}.zoutcorrs = zscore(distances{manip1, digit}.outcorrs);
        end
    end
end

%% Z-score by all correlation values
temp = zeros(size(distances, 1), size(distances, 2), size(distances{1, 1}.outcorrs, 2));

for manip1 = 1:8
    for digit = 1:10
        temp(manip1, digit, :) = distances{manip1, digit}.outcorrs;
    end
end

temp = reshape(temp, 1, size(temp, 1) * size(temp, 2) * size(temp, 3));
zscores = zscore(temp);
zscores = reshape(zscores, size(distances, 1), size(distances, 2), size(distances{1, 1}.outcorrs, 2));

for manip1 = 1:8
    for digit = 1:10
        distances{manip1, digit}.zscoredcorr = zscores(manip1, digit, :);
    end
end

%% Jaccard

%% PCA?

%% Plotting: ssimvals (comparing digits across manips)

ys = zeros(8, 10);
for i = 1:8
    subplot(2, 4, i)
    for j = 1:10
        ys(i, j) = distances{i, j}.ssimval;
    end
    plot(0:9, ys(i, :), '-o');
    title(['Mean SSIM by digit: ', maniplabels{i}]);
    xlabel('Digit')
    ylabel('SSIM value')
end

%% Plotting: ssimvals (comparing manips across digits)
ys = zeros(10, 8);
for i = 1:10
    figure(i)
    for j = 1:8
        ys(i, j) = distances{j, i}.ssimval;
    end
    bar(ys(i, :));
    title(['Mean SSIM by manipulation: ', int2str(i-1)]);
    xticklabels(maniplabels)
    xlabel('Manipulation')
    ylabel('SSIM value')
    ylim([0,1])
end

%% Plotting: all average values

% plot all average values
for i = 1:8
    figure(i)
    for j = 1:10
        subplot(2, 5, j)
        imagesc(distances{i, j}.avg)
        colormap gray
    end
end

%% Plotting: correlations (digit to digit)
for j = 1:8
    subplot(2, 4, j)
    corrmat = zeros(10);
    for i = 1:10
        corrmat(i, :) = distances{j, i}.incorrs; 
    end
    imagesc(corrmat)
    colormap(bluewhitered)
    xticks(0:10);
    yticks(0:10);
    xticklabels([1, 0:9]);
    yticklabels([1, 0:9]);
    title(['Digit Similarity: ', maniplabels{j}]);
    xlabel('Digit')
    ylabel('Digit')
    colorbar
    caxis([0, 1])
end

%% Plotting: correlations (dataset to dataset)
for i = 1:10
    im = figure(i);
    corrmat = zeros(8);
    for j = 1:8
        corrmat(j, :) = distances{j, i}.outcorrs;
    end
    imagesc(corrmat)
    colormap(bluewhitered)
    xticklabels(maniplabels);
    yticklabels(maniplabels);
    title(['Manip to manip correlation (mean image): ', int2str(i - 1)]);
    colorbar
    caxis([-1, 1])
end

%% Plotting: correlations (dataset to dataset, z-score per dataset, opt. abs val)

for i = 1:10
    im = figure(i);
    corrmat = zeros(8);
    for j = 1:8
        corrmat(j, :) = distances{j, i}.zoutcorrs;
    end
    % corrmat = abs(corrmat);
    imagesc(corrmat)
    colormap(bluewhitered)
    xticklabels(maniplabels);
    yticklabels(maniplabels);
    title(['Dataset to dataset mean image correlation z-scored (over dataset): ', int2str(i - 1)]);
    colorbar
    caxis([0, 1])
end

%% Plotting: correlations (dataset to dataset, z-score overall, opt. abs val)

for i = 1:10
    im = figure(i);
    corrmat = zeros(8);
    for j = 1:8
        corrmat(j, :) = distances{j, i}.zscoredcorr;
    end
    % abs val line
    % corrmat = abs(corrmat);
    imagesc(corrmat)
    colormap(bluewhitered)
    xticklabels(maniplabels);
    yticklabels(maniplabels);
    title(['Dataset to dataset mean image correlation z-scored (over all): ', int2str(i - 1)]);
    colorbar
    
    % regular
    % caxis([-1.9, 2])
    
    % absvalue
    % caxis([0, 1])
end

%% try clustering?

fulllinims = zeros(80, 28*28);
count = 1;
for i = 1:8
    for j = 1:10
        fulllinims(count, :) = distances{i, j}.linim;
        count = count + 1;
    end
end
%% dendrogram?
% treelbls = {'fliplr8 (19)', 'fliplr9 (20)', 'fliplr7 (18)', 'fliplr6 (17)', 'fliplr5 (16)'...
%     'fliplr3 (14)', 'orig1 (2)', 'fliplr0 (11)', 'fliplr4 (15)', 'orig2 (3)'...
%     'orig0 (1)', 'orig3 (4)', 'orig8 (9)', 'fliplr1 (12)', 'orig5 (6)', 'fliplr2 (13)'...
%     'orig7 (8)', 'orig9 (10)', 'orig6 (7)', 'flipud0 (21)', 'flipud2 (23)'...
%     'flipud3 (24)', 'flipud4(25)', 'flipud5 (26)', 'flipud6 (27)', 'flipud7 (28)'...
%     'flipud8 (29)', 'flipud9 (30)', 'flipud1 (22)'};
% treelbls = string();
% for i = 0:79
%     treelbls(i + 1) = int2str(i);
% end
dendrogram(linkage(clusterdata(fulllinims, 10), 'ward', 'Euclidean'), 0)

%%
% %% Plot out 16 images (testing)
% 
% for i = 1:16
%     subplot(4, 4, i);
%     imagesc(nums{1}(:, :, i));
%     colormap gray
% end
% 
% %% Mean image per digit
% meanims = zeros(28, 28, 10);
% for i = 1:10
%     meanims(:, :, i) = mean(nums{i}(:, :, :), 3);
%     
%     % adds/updates field in struct
% %     if isfield(distances{i}, 'oavg')
% %         distances{i} = rmfield(distances{i}, 'oavg');
% %     end
%     distances{i}.oavg = 255 - meanims(:, :, i);
% end
% 
% %% Difference
% diffim = zeros(28, 28, 10);
% for i = 1:10
%     diffim(:, :, i) = nums{i}(:, :, 1) - meanims(:, :, i);
%     
%     % plotting to check
%     subplot(2, 5, i)
%     imagesc(diffim(:, :, i))
%     colormap gray
%     
%     % adds/updates field in struct
% %     if isfield(distances{i}, 'diffIm1')
% %         distances{i} = rmfield(distances{i}, 'diffIm1');
% %     end
%     distances{i}.diffIm1 = diffim(:, :, i);
% end

%% SSIM
% ssimval = ssim(nums{1}(:, :, 1), meanims(:, :, 1));

% mean of each dataset to each other
% mean of each number within datasets
% etc. through jaccard and PCA

% redblue colormap from https://www.mathworks.com/matlabcentral/fileexchange/25536-red-blue-colormap