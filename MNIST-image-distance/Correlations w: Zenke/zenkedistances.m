%% Create cell array of structs
zenkesets = cell(5, 10);
for i = 1:10
    for j = 1:5
        zenkesets{j, i} = struct();
    end
end

%% Read csvs into structs
% t5.csv = MNIST basic (task 0), t1 = MNIST rotated (task 1), etc.
for n = 0:4
t = csvread(['t' int2str(n) '.csv']);
label = csvread(['labels' int2str(n) '.csv']);
t = squeeze(t);
    for j = 1:10
        zenkesets{n + 1, j}.set = t(logical(label(:, j)), :);
    end
end

%% calculate mean image for each digit (relinearize so it's the correct rotation)
for i = 1:5
    for j = 1:10
        zenkesets{i, j}.linim = mean(zenkesets{i, j}.set);
        
        zenkesets{i, j}.linim = reshape(reshape(zenkesets{i, j}.linim, [28, 28])', [1, 28*28]);
    end
end

%% Subtract images from 255: inverts to correct images
for i = 1:5
    for j = 1:10
        zenkesets{i, j}.linim = 255 - zenkesets{i, j}.linim;
    end
end

%% concatenate our datasets and their datasets
d = [distances; zenkesets];

%% Calculate dataset-dataset correlations for each image
for i = 1:13
    for j = 1:10
        d{i, j}.outcorrs = zeros(1, 10);
        for k = 1:13
            d{i, j}.outcorrs(k) = corr(d{i, j}.linim', d{k, j}.linim');
        end
    end
end

%% Find mean correlation over all digits
meancorrs = zeros(13, 13);
for i = 1:13
    for j = 1:10
        meancorrs(i, :) = meancorrs(i, :) + d{i, j}.outcorrs;
    end
end
meancorrs = meancorrs / 13;

%% Plot correlation matrix for each digit

maniplabels = {'Original', 'Fliplr', 'Flipud', 'Rot90', 'Inverse', 'Checkerboard', 'Invbot', 'Swapud', 'Zenke1', 'Zenke2', 'Zenke3', 'Zenke4', 'Zenke5'};


for i = 1:10
    im = figure(i);
    corrmat = zeros(13);
    for j = 1:13
        corrmat(j, :) = d{j, i}.outcorrs;
    end
    imagesc(corrmat)
    colormap(bluewhitered)
    xticklabels(maniplabels);
    xticks(1:13)
    yticks(1:13)
    yticklabels(maniplabels);
    title(['Dataset to Dataset correlation (mean image): ', int2str(i - 1)]);
    colorbar
    caxis([-1, 1])
    xlabel('Dataset')
    ylabel('Dataset')
end

%% plot meancorrs
maniplabels = {'Original', 'Fliplr', 'Flipud', 'Rot90', 'Inverse', 'Checkerboard', 'Invbot', 'Swapud', 'Zenke1', 'Zenke2', 'Zenke3', 'Zenke4', 'Zenke5'};

    figure(1);
    imagesc(meancorrs)
    colormap(bluewhitered)
    xticklabels(maniplabels);
    xticks(1:13)
    yticks(1:13)
    yticklabels(maniplabels);
    title('Dataset to Dataset mean correlation over all digits');
    colorbar
    caxis([-1, 1])
    xlabel('Dataset')
    ylabel('Dataset')