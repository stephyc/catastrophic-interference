%% Bar graph accuracies

xs = 1:7;
ys = [.9644, .9566, .9419, .9305, .8895, .9356, .9206];
b = bar(ys, 'FaceColor', 'flat');
barcmap = jet(length(ys));
figure (1)
for i = 1:length(ys)
    b(i) = bar(xs(i), ys(i));
    set(b(i), 'FaceColor', barcmap(i,:));
    hold on
end
title('Results: Training and Testing on Each Dataset');
xlabel('Image Mutation');
ylabel('Accuracy');
set(gca, 'XTick', xs);
set(gca, 'XTickLabel', {'Rotate 90 degrees' 'Invert' 'Flip Left-Right' 'Swap Top-Bottom' 'Checkerboard' 'Flip Up-Down'...
    'Invert Bottom'});

% Rot90: {'accuracy': 0.9644, 'loss': 0.114561394, 'global_step': 2401}
% Inv: {'accuracy': 0.9566, 'loss': 0.1464758, 'global_step': 2601}
% fliplr: {'accuracy': 0.9419, 'loss': 0.2036909, 'global_step': 2801}
% cutud: {'accuracy': 0.9305, 'loss': 0.2341149, 'global_step': 3001} 
% checkerboard: {'accuracy': 0.8895, 'loss': 0.37382987, 'global_step': 3201} 
% flipud: {'accuracy': 0.9356, 'loss': 0.21395998, 'global_step': 3601}
% invbot: {'accuracy': 0.9206, 'loss': 0.25648096, 'global_step': 3801}

%% Original --> Rot90

% trained/tested: original
epochs = 1:5;
figure(1)
y1 = [.979, .9778, .9816, .9834, .9854];
plot(epochs, y1, '-o');
xlim([1, 5]);
ylim([.25, 1]);
title('Initial training: Original set')
xlabel('Epoch');
ylabel('Accuracy')
legend('Original')

% trained/tested: rot90, original/rot90
figure(2)
y2rot = [.9066, .9673, .9749, .9792, .9819];
y2orig = [.7349, .5208, .5220, .5056, .4279];
plot(epochs, [y2rot; y2orig], '-o');
hold on
xlim([1, 5]);
ylim([.25, 1]);
title('Second training: Rotated 90 set')
xlabel('Epoch');
ylabel('Accuracy')
legend('Rotated 90', 'Original')