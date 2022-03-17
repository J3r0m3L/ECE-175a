clear
load("data.mat");
load("label.mat");


%------------------- Correct Sample Mean ----------------
% calculate the mean for the Guassian
% Literally just saw that its independent zero mean Guassian
sample_mean = zeros(784, 10);
sample_meanDenominator = zeros(1, 10);
for i = 0:9
    current_indexes = find(labelTrain == i);
    sample_meanDenominator(i + 1) = length(current_indexes);
    sample_mean(:, i + 1) = sum(reshaped_imageTrain(:, current_indexes), 2) / sample_meanDenominator(i + 1);
end
%---------------------------------------------------------




% Reshape Arrays into 784 x 500/5000
reshaped_imageTrain = reshape(imageTrain, 784, 5000);
reshaped_imageTest = reshape(imageTest, 784, 500);

train_dims = size(reshaped_imageTrain);
test_dims = size(reshaped_imageTest);

% Question 1.
class_length = zeros(1, 10);

sample_mean = zeros(784, 10);
sample_denominator = sum(reshaped_imageTrain, 2);
sample_denominator(sample_denominator == 0) = 1;
for i = 0:9
    iteration_index = find(labelTrain == i);
    class_length(i + 1) = length(iteration_index);
    sample_mean(:, i + 1) = sum(reshaped_imageTrain(:, iteration_index), 2) ./ sample_denominator;
end
reshaped_sampleMean = reshape(sample_mean, 28, 28, 10);
%imshow(reshaped_sampleMean(:, :, 8));

% Question 2.
alpha = -2 * log10(class_length ./ train_dims(2));
norms = zeros(500, 10);
for i = 0:9
    repeated_sampleMean = repmat(sample_mean(:, i + 1), 1, test_dims(2));
    norms(:, i + 1) = transpose(sum((reshaped_imageTest - repeated_sampleMean) .^ 2));
end

[throw_away ,predicted_labels] = min(norms + repmat(alpha, test_dims(2), 1), [], 2);
predicted_labels = predicted_labels - 1;


% Error Given Class
given_classError = zeros(1, 10);
for i = 0:9
    iteration_index = find(labelTest == i);
    given_size = length(iteration_index);
    pruned_labelTest = labelTest(iteration_index);
    pruned_predictedLabels = predicted_labels(iteration_index);
    given_classError(i + 1) = length(nonzeros(pruned_labelTest - pruned_predictedLabels)) / given_size;
end


% Total Error
total_Error = length(nonzeros(labelTest - predicted_labels)) / test_dims(2);

% Extra Credit
lmatrix = cov(transpose(sample_mean));
imshow(lmatrix, []);