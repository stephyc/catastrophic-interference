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
title('Results: Training and Testing on Each Dataset', 'FontSize', 50);
xlabel('Image Mutation', 'FontSize', 80);
ylabel('Accuracy', 'FontSize', 80);
set(gca, 'XTick', xs);
set(gca, 'XTickLabel', {'Rotate 90' 'Invert' 'Flip LR' 'Swap TB' 'Checkerboard' 'Flip UD'...
    'Invert Bottom'}, 'FontSize', 20);

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
plot(epochs, y1, '-o', 'LineWidth', 7, 'MarkerSize', 13);
xlim([1, 5]);
ylim([.25, 1]);
title('Initial training: Original set', 'FontSize', 50)
xlabel('Epoch', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40)
set(gca, 'FontSize', 30)
legend('Original')

% trained/tested: rot90, original/rot90
figure(2)
y2rot = [.9066, .9673, .9749, .9792, .9819];
y2orig = [.7349, .5208, .5220, .5056, .4279];
plot(epochs, [y2orig; y2rot], '-o', 'LineWidth', 7, 'MarkerSize', 13);
hold on
xlim([1, 5]);
ylim([.25, 1]);
title('Second training: Rotated 90 set', 'FontSize', 50)
xlabel('Epoch', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40)
set(gca, 'FontSize', 30);
legend('Original', 'Rotated 90')

%% Allie's data
barcmap = jet(4);

% Training set 1
xs = 1;
y1 = .9867;
b = bar(y1, 'FaceColor', 'flat');
figure(1)
for i = 1:length(y1)
    b(i) = bar(xs(i), y1(i));
    set(b(i), 'FaceColor', barcmap(i,:));
    hold on
end
title('Training Round 1: Original', 'FontSize', 50);
xlabel('Image Mutation', 'FontSize', 40);
ylim([0, 1]);
ylabel('Accuracy', 'FontSize', 40);
set(gca, 'XTick', xs);
set(gca, 'XTickLabel', {'Original'}, 'FontSize', 20);
hold off
%% Allie's Data 2
% original, 90, inv, flip ud

% Training set 2
xs = 1:2;
y2 = [.05064, .9881];
b = bar(y2, 'FaceColor', 'flat');
figure(2)
for i = 1:length(y2)
    b(i) = bar(xs(i), y2(i));
    set(b(i), 'FaceColor', barcmap(i,:));
    hold on
end
title('Training Round 2: Rotation 90', 'FontSize', 50);
xlabel('Image Mutation', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40);
set(gca, 'XTick', xs);
set(gca, 'XTickLabel', {'Original' 'Rotate 90'}, 'FontSize', 20);
%% Allie's Data 3

% Training set 3
xs = 1:3;
y3 = [.75524, .3679, .9905];
b = bar(y3, 'FaceColor', 'flat');
figure (3)
for i = 1:length(y3)
    b(i) = bar(xs(i), y3(i));
    set(b(i), 'FaceColor', barcmap(i,:));
    hold on
end
title('Training Round 3: Inversion', 'FontSize', 50);
xlabel('Image Mutation', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40);
set(gca, 'XTick', xs);
set(gca, 'XTickLabel', {'Original' 'Rotate 90' 'Inversion'}, 'FontSize', 20);

%% Allie's Data 4
% Training set 4
xs = 1:4;
y4 = [.2152, .2149, .468, .9403];
b = bar(y4, 'FaceColor', 'flat');
figure(4)
for i = 1:length(y4)
    b(i) = bar(xs(i), y4(i));
    set(b(i), 'FaceColor', barcmap(i,:));
    hold on
end
title('Training Round 4: Flip Up-Down', 'FontSize', 50);
xlabel('Image Mutation', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40);
set(gca, 'XTick', xs);
set(gca, 'XTickLabel', {'Original' 'Rotate 90' 'Inversion' 'Flip-UD'}, 'FontSize', 20);

%% Stephanie's Data

epochs = 1:5;
figure(1)
acc_original =[0.8888,0.9697,0.9772,0.9814,0.9827];
plot(epochs, acc_original, '-o', 'LineWidth', 7, 'MarkerSize', 13);
xlim([1, 5]);
ylim([.25, 1]);
title('Training Round 1: Original', 'FontSize', 50)
xlabel('Epoch', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40)
set(gca, 'FontSize', 30)
legend('Original')

figure(2)
acc_rot90 =[0.9047,0.9674,0.9754,0.9793,0.9824]; % acc
acc_original_2 =[0.7928,0.6905,0.6029,0.5139,0.4359]; % val_acc
plot(epochs, [acc_original_2; acc_rot90], '-o', 'LineWidth', 7, 'MarkerSize', 13);
xlim([1, 5]);
ylim([0, 1]);
title('Training Round 2: Rotated 90', 'FontSize', 50)
xlabel('Epoch', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40)
set(gca, 'FontSize', 30)
legend('Original', 'Rotated 90')

figure(3)
acc_3 = [0.9419,0.9741,0.9805,0.9839,0.9859];
acc_original_3 = [0.8580,0.7565,0.8086,0.7647,0.7626];
plot(epochs, [acc_original_3; acc_3], '-o', 'LineWidth', 7, 'MarkerSize', 13);
xlim([1, 5]);
ylim([0, 1]);
title('Training Round 3: ', 'FontSize', 50)
xlabel('Epoch', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40)
set(gca, 'FontSize', 30)
legend('Original', 'Inversion')

figure(4)
acc_4 = [0.9121,0.9696,0.9761,0.9802,.9830];
acc_original_4 = [0.2707,0.2293,0.2181,0.1811,0.1953];
plot(epochs, [acc_original_4; acc_4], '-o', 'LineWidth', 7, 'MarkerSize', 13);
xlim([1, 5]);
ylim([0, 1]);
title('Training Round 4: ', 'FontSize', 50)
xlabel('Epoch', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40)
set(gca, 'FontSize', 30)
legend('Original', 'Flip-UD')

figure(5)
acc_5 = [0.9058,0.9681,0.9765,0.9797,0.9824];
acc_original_5 = [0.1322,0.1702,0.1506,0.1660,0.1096];
plot(epochs, [acc_original_5; acc_5], '-o', 'LineWidth', 7, 'MarkerSize', 13);
xlim([1, 5]);
ylim([0, 1]);
title('Training Round 4: ', 'FontSize', 50)
xlabel('Epoch', 'FontSize', 40);
ylabel('Accuracy', 'FontSize', 40)
set(gca, 'FontSize', 30)
legend('Original', 'Round 5')