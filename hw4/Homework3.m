clear
load("dataNew.mat");

noisy_image = double(imread("sampletest.png"));
comparable_image = double(imread("sampletrain.png"));

reshaped_imageTrain = reshape(imageTrain, 784, 5000) / 255;
reshaped_imageTest = reshape(imageTestNew, 784, 500) / 255;
reshaped_noisyImage = reshape(noisy_image, 784, 1) / 255;
reshaped_comparableImage = reshape(comparable_image, 784, 1) / 255;

% calcualte alpha
lauv2 = transpose(reshaped_noisyImage) * reshaped_comparableImage;
lauv3 = transpose(reshaped_comparableImage) * reshaped_comparableImage;
alpha = lauv2 / lauv3;
lauv = reshaped_noisyImage .* reshaped_comparableImage ./ reshaped_comparableImage .^ 2;