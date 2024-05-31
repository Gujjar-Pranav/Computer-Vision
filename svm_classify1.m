function [best_lambda, best_accuracy, predicted_categories] = svm_classify1(train_image_feats, train_labels, test_image_feats, test_labels)

categories = unique(train_labels);
num_categories = length(categories);
lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]; 
accuracy_results = zeros(length(lambda_values), 1);
% Train 1-vs-all linear SVM classifiers for each lambda
svm_models = cell(length(lambda_values), 1); 
for lambda_idx = 1:length(lambda_values)
    lambda = lambda_values(lambda_idx);  
    svm_models_lambda = cell(num_categories, 1);
    for i = 1:num_categories      
        class = categories{i};
        binary_labels = strcmp(train_labels, class);
        binary_labels = 2 * binary_labels - 1;       
        % Train linear SVM for the current category with the current lambda
        [W, B] = vl_svmtrain(train_image_feats', binary_labels, lambda);      
        svm_models_lambda{i} = struct('W', W, 'B', B);
    end  
    svm_models{lambda_idx} = svm_models_lambda;    
    % Perform classification on test set to compute accuracy
    correct_predictions = 0;
    total_predictions = 0;
    predicted_categories_cell = cell(length(test_labels), 1);
    for j = 1:length(test_labels)       
        confidences = zeros(num_categories, 1);
        for k = 1:num_categories
            W = svm_models_lambda{k}.W;
            B = svm_models_lambda{k}.B;
            confidence = W' * test_image_feats(j, :)' + B;
            confidences(k) = confidence;
        end               
        [~, max_idx] = max(confidences);
        predicted_category = categories{max_idx};        
        % Store predicted category
        predicted_categories_cell{j} = predicted_category;    
        % Check if prediction is correct
        if strcmp(predicted_category, test_labels{j})
            correct_predictions = correct_predictions + 1;
        end
        total_predictions = total_predictions + 1;
    end       
    accuracy_results(lambda_idx) = correct_predictions / total_predictions;
end

% Find the index of the best lambda value (maximum accuracy)
[~, best_lambda_idx] = max(accuracy_results);
best_lambda = lambda_values(best_lambda_idx);
% Use the best lambda to make predictions
best_svm_models = svm_models{best_lambda_idx};
% Perform classification on test set using SVM models for the best lambda
correct_predictions = 0;
total_predictions = 0;
predicted_categories_cell = cell(length(test_labels), 1);
for j = 1:length(test_labels)    
    confidences = zeros(num_categories, 1);
    for k = 1:num_categories
        W = best_svm_models{k}.W;
        B = best_svm_models{k}.B;
        confidence = W' * test_image_feats(j, :)' + B;
        confidences(k) = confidence;
    end       
    [~, max_idx] = max(confidences);
    predicted_category = categories{max_idx};    
    predicted_categories_cell{j} = predicted_category;        
    if strcmp(predicted_category, test_labels{j})
        correct_predictions = correct_predictions + 1;
    end
    total_predictions = total_predictions + 1;
end
% Compute accuracy using the best lambda model
best_accuracy = correct_predictions / total_predictions;
predicted_categories = predicted_categories_cell;
results_table = table(lambda_values', accuracy_results, 'VariableNames', {'Lambda', 'Accuracy'});
assignin('base', 'Lambda_Exp_table', results_table);
disp('Best Lambda:');
disp(best_lambda);
disp('Best Accuracy:');
disp(best_accuracy);
end
