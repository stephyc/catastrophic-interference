function [] = savemnist(var)

for i = 1:size(var, 3)
    im = mat2gray(var(:, :, i));
    imname = ['rot90', num2str(i), '.png'];
    imwrite(im, imname); 
end