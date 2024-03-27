% gt = imread('abd_ct_gt.png')
% im = imread('abd_ct_img.png')
% figure
% subplot(1,3,1), imshow(gt)
% subplot(1,3,2), imshow(im)

% img 原始图像
% label 目标标签
% color 每个类的颜色
% alpha 每个类颜色的不透明度
function colorimg = drawlabel2image(img, label, color, alpha)
[row, col, dim] = size(img);
if dim==1
    img = cat(3, img, img, img);
elseif dim==3
else
    error('请输入灰度图或RGB图')
end

% 预处理
img = im2double(img);
label = uint16(label);
nlabel = max(label(:));

% color修正
while size(color,1)<nlabel
    color = cat(1,color,color);
end

if size(color,1)>2*nlabel
    gap = floor(size(color,1)/nlabel);
    color = color(1:gap:end,:);
end

alpha = alpha(:);
while length(alpha)<nlabel
    alpha = cat(1, alpha, alpha);
end

% 保留背景
mask = double(label>0);
bg = img.*double(~mask);


obj = zeros(row,col,3,nlabel);
for idx = 1:nlabel
    objmask = double(label==idx);    
    R = img(:,:,1).*objmask*(1-alpha(idx))+objmask*color(idx,1)*alpha(idx);
    G = img(:,:,2).*objmask*(1-alpha(idx))+objmask*color(idx,2)*alpha(idx);
    B = img(:,:,3).*objmask*(1-alpha(idx))+objmask*color(idx,3)*alpha(idx);
    obj(:,:,:,idx) = cat(3,R,G,B);
end

colorimg = sum(obj,4)+bg;

end


