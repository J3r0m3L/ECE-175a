clear
load("dataNew.mat");
noisy_image = double(imread("sampletest.png"));
comparable_image = double(imread("sampletrain.png"));

reshaped_imageTrain = reshape(imageTrain, 784, 5000) / 255;
reshaped_imageTest = reshape(imageTestNew, 784, 500) / 255;
reshaped_noisyImage = reshape(noisy_image, 784, 1) / 255;
reshaped_comparableImage = reshape(comparable_image, 784, 1) / 255;

% calculate variance for the sample_images
variance = zeros(784, 10);
class_testLength = zeros(1, 10);
for i = 0:9
    current_trainIndexes = find(labelTrain == i);
    current_testIndexes = find(labelTestNew == i);
    
    class_testLength(i + 1) = length(current_testIndexes);
    current_trainIndexes = current_trainIndexes(1:class_testLength(i+1));
    
    term_one = class_testLength(i + 1) .* sum((reshaped_imageTrain(:, current_trainIndexes) .^ 2), 2);
    term_two = sum(reshaped_imageTrain(:, current_trainIndexes), 2) .^ 2;
    variance(:, i + 1) = term_one - term_two;
end

% calculate covariance for the sample_images
covariance = zeros(784, 10);
for i = 0:9
    current_trainIndexes = find(labelTrain == i);
    current_testIndexes = find(labelTestNew == i);
    
    current_trainIndexes = current_trainIndexes(1:class_testLength(i+1));
    term_one = class_testLength(i+1) .* sum((reshaped_imageTrain(:, current_trainIndexes) .* reshaped_imageTest(:, current_testIndexes)), 2);
    term_two = sum(reshaped_imageTrain(:, current_trainIndexes), 2) .* sum(reshaped_imageTest(:, current_testIndexes), 2);
    covariance(:, i + 1) = term_one - term_two;
end

variance(variance == 0) = 1;
alpha = covariance ./ variance;

N = reshaped_noisyImage - alpha(:, 4) .* reshaped_comparableImage;

least_SquaresDistances = inv(transpose(alpha) * alpha) * transpose(alpha) * reshaped_imageTest;
[throw_away, predicted_labels] = min(least_SquaresDistances);

given_classError = zeros(1, 10);
for i = 0:9
    given_Index = find(labelTestNew == i);
    given_Size = length(given_Index);
    pruned_labelTest = labelTestNew(given_Index);
    pruned_predictedLabels = transpose(predicted_labels(given_Index));
    given_classError(i + 1) = length(nonzeros(pruned_labelTest - pruned_predictedLabels)) / given_Size;
end

total_Error = length(nonzeros(labelTestNew - transpose(predicted_labels))) / 500;



