clear;
% filename='BinaryNet_BinaryWeights_0.75.h5'; 
% h5disp(filename);
S = load('val.mat');
pic = imread('6_107.jpg');
image = uint8(zeros(32, 32, 3));
% pic = pic / 255;
image_size = size(pic);
if length(image_size) == 2
  image(:,:,1) = pic;
  image(:,:,2) = pic;
  image(:,:,3) = pic;
else
      imag = rgb2gray(pic);
      image(:,:,1) = uint8(imag);
      image(:,:,2) = uint8(imag);
      image(:,:,3) = uint8(imag);
%   image = pic;
end

%conv1
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv1/conv1/kernel:0');
image = conv(image, weights);

%bn1
x = S.bn1.x;
image = bn_m(image, x);

%conv2
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv2/conv2/kernel:0');
image = conv(image, weights);

%pool2
image = max_pooling(image, 2);
image(16,16,:);

%bn2
x = S.bn2.x;
image = bn_m(image, x);

%conv3
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv3/conv3/kernel:0');
image = conv(image, weights);

%bn3
x = S.bn3.x;
image = bn_m(image, x);

%conv4
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv4/conv4/kernel:0');
image = conv(image, weights);

%pool4
image = max_pooling(image, 2);

%bn4
x = S.bn4.x;
image = bn_m(image, x);

%conv5
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv5/conv5/kernel:0');
image = conv(image, weights);

%bn5
x = S.bn5.x;
image = bn_m(image, x);

%conv6
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv6/conv6/kernel:0');
image = conv(image, weights);

%pool6
image = max_pooling(image, 2);

%bn6
x = S.bn6.x;
image = bn_m(image, x);

image = flatten(image);

%binary_dense1
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/dense5/dense5/kernel:0');
image = full_connect(image, weights);

%bn7
x = S.bn7.x;
image = fc_bn_m(image, x);

%binary_dense2
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/dense6/dense6/kernel:0');
image = full_connect(image, weights);

%bn8
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/moving_variance:0');
image = fc_bn(image, beta, gamma, tmean, tvariance)
% x = S.bn8.x;
% image = fc_bn_m(image, x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%conv
function [image_conv_out] = conv(image, weights)
  weights = permute(weights,[4, 3, 2, 1]);
  [~, ~, ~, filters] = size(weights);  
  weights = rot90(weights,2);

  image_size = size(image);
  channels = image_size(3);
  image_size = image_size(1);
  
  image_conv_out = zeros(image_size,image_size,filters);
  for filter_conv = 1:filters
    for image_channels = 1:channels
      con = conv2(image(:,:,image_channels),weights(:,:,image_channels,filter_conv), 'same');
      image_conv_out(:,:,filter_conv) = image_conv_out(:,:,filter_conv) + con;
    end
  end
end

%bn_merge
function [image] = bn_m(image, x)
 [~, ~, filters] = size(image);
 for ch = 1:filters
    b = image(:, :, ch);
    b(image(:, :, ch) <= x(ch)) = -1; 
    b(image(:, :, ch) >  x(ch)) = 1;
    image(:, :, ch) = b;
 end
end

%fc_bn _merge
function [image] = fc_bn_m(image, x)
 [~, filters] = size(image);
 for ch = 1:filters
    b = image(:, ch);
    b(image(:, ch) <= x(ch)) = -1;
    b(image(:, ch) >  x(ch)) =  1;
    image(:, ch) = b;
 end
end

%max_pooling
function layer1_mp = max_pooling(layer1,ks)
  [H,W,C] = size(layer1);
  Hp = ceil(H / ks);
  Wp = ceil(W / ks) ;
  chan = ones(Hp,Wp)*NaN;
  layer1_mp = zeros(Hp,Wp,C);
  for c = 1:C
      chan(1:H,1:W) = layer1(:,:,c);
      bx = 1:ks;
      by = 1:ks;
      
      for y = 1:Hp
          for x = 1:Wp
              A = chan(by,bx);
              layer1_mp(y,x,c) = max(A(:));
              if bx < H
                bx = bx + 2;
              end
          end
          
          bx = 1:ks;
          if by < W
            by = by + 2;
          end
      end    
  end
end

function [image] = fc_bn(image, beta, gamma, tmean, tvariance)
 [~, filters] = size(image);
 eps = 1.0e-8;
 for ch = 1:filters
    image(1, ch) = (image(1, ch) - tmean(ch)) / (sqrt(tvariance(ch) + eps));
    image(1, ch) = gamma(ch) * image(1, ch) + beta(ch);
  end
end

%flatten
function [image] = flatten(image)
  image = permute(image, [2, 3, 1]);
  [~, ~, size_img] = size(image);
  img = zeros([512, 4, 4]);
  for filter = 1:size_img
    img(:, :, filter) = image(:, :, filter)';
  end
    image = img(:);
end

%full_connected_layer
function [full_out] = full_connect(image, weights)
  size_weight = size(weights);
  kernels = size_weight(1);
  full_out = zeros(1, kernels);
  for full_size = 1:kernels
    full_out(1, full_size) = dot(image, weights(full_size, :));
  end
end


