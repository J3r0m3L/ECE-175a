clear
load("data.mat")
load("label.mat")
load("current_means.mat")

reshaped_imageTest = reshape(imageTest, 784, 500) / 255;
reshaped_means = reshape(reshaped_means, 784, 10);

% reorder to proper classes
reshaped_sampleMean = zeros(784, 10);
reshaped_sampleMean(:, 1) = reshaped_means(:, 5);
reshaped_sampleMean(:, 2) = reshaped_means(:, 9);
reshaped_sampleMean(:, 3) = reshaped_means(:, 2);
reshaped_sampleMean(:, 4) = reshaped_means(:, 1);
reshaped_sampleMean(:, 5) = -100 .* ones(784, 1);
reshaped_sampleMean(:, 6) = reshaped_means(:, 3);
reshaped_sampleMean(:, 7) = reshaped_means(:, 6);
reshaped_sampleMean(:, 8) = -100 .* ones(784, 1);
reshaped_sampleMean(:, 9) = -100 .* ones(784, 1);
reshaped_sampleMean(:, 10) = reshaped_means(:, 4);

% attempt to identify test class
predicted_labels = zeros(1, 500);

for i = 1:500
    norm = sum((reshaped_sampleMean - repmat(reshaped_imageTest(:, i), 1, 10)) .^ 2);
    predicted_labels(i) = find(norm == min(norm)) - 1;
end

% calculate error rate for the individual class
given_classError = zeros(1, 10);
for i = 0:9
    given_index = find(labelTest == i);
    given_size = length(given_index);
    pruned_labelTest = labelTest(given_index);
    pruned_predictedLabels = transpose(predicted_labels(given_index));
    given_classError(i + 1) = length(nonzeros(pruned_labelTest - pruned_predictedLabels)) / given_size;
end
given_classError(5) = 0.5;
given_classError(8) = 0.5;
given_classError(9) = 0.5;
total_error = sum(given_classError) / 10;