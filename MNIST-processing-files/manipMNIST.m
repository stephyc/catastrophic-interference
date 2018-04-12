function newims = manipMNIST(var)

newims = var;

% % flip lr
% newims = fliplr(var);

% % upside down
% newims = flipud(var);

% rotate direction 1
newims = imrotate(var, 90);



% for i = 1:size(var, 3)
%     
%     % checkerboard
% %     x = 1:2:28;
% %     y = 1:2:28;
% %     newims(x, y, i) = 255 - newims(x, y, i);
% %     x = 2:2:28;
% %     y = 2:2:28;
% %     newims(x, y, i) = 255 - newims(x, y, i);
%     
%     % inv bottom
% %     x = 1:14;
% %     y = 1:28;
% %     newims(x, y, i) = 255 - var(x, y, i);
%     
%     % cut top bottom
% %     x = 1:14;
% %     y = 1:28;
% %     z = 15:28;
% %     newims(x, y, i) = var(z, y, i);
% %     newims(z, y, i) = var(x, y, i);
% %     
% %     
% %     
% %     
% %     % rotated direction 2
% %     newims = imrotate(var, 270);
%     
% end
% 
% end