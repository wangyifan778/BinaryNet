clear;
filename='BinaryNet_BinaryWeights_0.75.h5'; 
h5disp(filename);

pic = imread('5_168.jpg');
pic = single(pic);
% pic = pic / 255;
image_size = size(pic);
if length(image_size) == 2
  image(:,:,1) = pic;
  image(:,:,2) = pic;
  image(:,:,3) = pic;
else
    image = pic;
end

%conv1
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv1/conv1/kernel:0');
image = conv(image, weights);

%bn1
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn1/bn1/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn1/bn1/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn1/bn1/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn1/bn1/moving_variance:0');
[image, X, ~] = bn_m1(image, beta, gamma, tmean, tvariance);
bn1 = struct('x',X);
save('val.mat', 'bn1');

%conv2
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv2/conv2/kernel:0');
image = conv(image, weights);

%pool2
image = max_pooling(image, 2);
image(16,16,:);

%bn2
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn2/bn2/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn2/bn2/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn2/bn2/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn2/bn2/moving_variance:0');
[image, X, ~] = bn_m(image, beta, gamma, tmean, tvariance);
bn2 = struct('x',X);
save('val.mat', 'bn2', '-append');

%conv3
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv3/conv3/kernel:0');
image = conv(image, weights);

%bn3
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn3/bn3/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn3/bn3/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn3/bn3/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn3/bn3/moving_variance:0');
[image, X, ~] = bn_m(image, beta, gamma, tmean, tvariance);
bn3 = struct('x',X);
save('val.mat', 'bn3', '-append');

%conv4
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv4/conv4/kernel:0');
image = conv(image, weights);

%pool4
image = max_pooling(image, 2);

%bn4
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn4/bn4/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn4/bn4/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn4/bn4/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn4/bn4/moving_variance:0');
[image, X, ~] = bn_m(image, beta, gamma, tmean, tvariance);
bn4 = struct('x',X);
save('val.mat', 'bn4', '-append');

%conv5
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv5/conv5/kernel:0');
image = conv(image, weights);

%bn5
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn5/bn5/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn5/bn5/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn5/bn5/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn5/bn5/moving_variance:0');
[image, X, ~] = bn_m(image, beta, gamma, tmean, tvariance);
bn5 = struct('x',X);
save('val.mat', 'bn5', '-append');

%conv6
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/conv6/conv6/kernel:0');
image = conv(image, weights);

%pool6
image = max_pooling(image, 2);

%bn6
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn6/bn6/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn6/bn6/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn6/bn6/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn6/bn6/moving_variance:0');
[image, X, ~] = bn_m(image, beta, gamma, tmean, tvariance);
bn6 = struct('x',X);
save('val.mat', 'bn6', '-append');

%flatten
image = flatten(image);

%binary_dense1
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/dense5/dense5/kernel:0');
image = full_connect(image, weights);

%bn7
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn7/bn7/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn7/bn7/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn7/bn7/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn7/bn7/moving_variance:0');
[image, X, S] = fc_bn_m(image, beta, gamma, tmean, tvariance);
bn7 = struct('x',X);
save('val.mat', 'bn7', '-append');

%binary_dense2
weights = h5read('BinaryNet_BinaryWeights_0.75.h5','/dense6/dense6/kernel:0');
image = full_connect(image, weights);

%bn8
beta = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/beta:0');
gamma = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/gamma:0');
tmean = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/moving_mean:0');
tvariance = h5read('BinaryNet_BinaryWeights_0.75.h5','/bn8/bn8/moving_variance:0');
[image, k, b] = line(image, beta, gamma, tmean, tvariance)
bn8 = struct('k',k, 'b',b);
save('val.mat', 'bn8', '-append');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%function%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%bn8
function [image, k, b] = line(image, beta, gamma, tmean, tvariance)
  [~, filters] = size(image);
  eps = 1.0e-8;
  k = zeros(filters, 1);
  b = zeros(filters, 1);
  for ch = 1:filters
      k(ch) = gamma(ch) / sqrt(tvariance(ch) + eps);
      b(ch) = beta(ch) - gamma(ch) *tmean(ch) / sqrt(tvariance(ch) + eps);
  end
  for ch = 1: filters
      image(1,ch) = k(ch) * image(ch) + b(ch);
  end
end
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
function [image, X, S] = bn_m(image, beta, gamma, tmean, tvariance)
 [~, ~, filters] = size(image);
 eps = 1.0e-8;
 X = zeros(filters,1);
 S = zeros(filters,1);
 for ch = 1:filters
    X(ch) = - beta(ch) * sqrt(tvariance(ch) + eps) / gamma(ch) + tmean(ch);
        X(ch) = floor(X(ch));
    b = image(:, :, ch);
    if gamma > 0
        S(ch) = 1;
    elseif gamma < 0 
        S(ch) = 0;
        
    end
    if S(ch) == 1
        b(image(:, :, ch) <= X(ch)) = -1; 
        b(image(:, :, ch) >  X(ch)) = 1;
    elseif gamma < 0 
        b(image(:, :, ch) >= X(ch)) = -1;
        b(image(:, :, ch) <  X(ch)) = 1;
    end
    image(:, :, ch) = b;
 end
end

function [image, X, S] = bn_m1(image, beta, gamma, tmean, tvariance)
 [~, ~, filters] = size(image);
 eps = 1.0e-8;
 X = zeros(filters,1);
 S = zeros(filters,1);
 for ch = 1:filters
    X(ch) = - beta(ch) * sqrt(tvariance(ch) + eps) / gamma(ch) + tmean(ch);
    X(ch) = X(ch) * 255.0;
    X(ch) = floor(X(ch));
    b = image(:, :, ch);
    if gamma > 0
        S(ch) = 1;
    elseif gamma < 0 
        S(ch) = 0;
    end
    if S(ch) == 1
        b(image(:, :, ch) <= X(ch)) = -1; 
        b(image(:, :, ch) >  X(ch)) = 1;
    elseif gamma < 0 
        b(image(:, :, ch) >= X(ch)) = -1;
        b(image(:, :, ch) <  X(ch)) = 1;
    end
    image(:, :, ch) = b;
 end
end

%fc_bn _merge
function [image, X, S] = fc_bn_m(image, beta, gamma, tmean, tvariance)
 [~, filters] = size(image);
 eps = 1.0e-8;
 X = zeros(filters,1);
 S = zeros(filters,1);
 for ch = 1:filters
    X(ch) = - beta(ch) * sqrt(tvariance(ch) + eps) / gamma(ch) + tmean(ch);
    X(ch) = floor(X(ch));
    b = image(:, ch);
    if gamma > 0
        S(ch) = 1;
    elseif gamma < 0 
        S(ch) = 0;
    end
    if S(ch) == 1
        b(image(:, ch) <= X(ch)) = -1;
        b(image(:, ch) >  X(ch)) =  1;
    elseif gamma < 0 
        b(image(:,ch) >= X(ch)) = -1;
        b(image(:,ch) <  X(ch)) =  1;
    end
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


