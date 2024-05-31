function [predicted_categories, best_k, best_accuracy, k_accuracy_table] = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, test_labels, max_k)
    % Check if max_k is provided, prompt user if not
    if nargin < 5 || isempty(max_k) || ~isnumeric(max_k)
        max_k = input('Enter the maximum value of k for nearest neighbor classification: '); % Prompt for k if not provided
    end

    % Check if max_k is numeric
    if ~isnumeric(max_k)
        error('Max_k must be a numeric value.');
    end

    % Calculate number of test images
    num_test = size(test_image_feats, 1);
    
    % Compute pairwise distances between test and train image features
    distances = pdist2(test_image_feats, train_image_feats);

    % Allocate space for accuracies for odd k values only
    all_accuracies = NaN(ceil(max_k / 2), 1);
    
    % Adjust to store odd k values only
    k_values = (1:2:max_k)';

    % Initialize best accuracy and best k
    best_accuracy = 0;
    best_k = 1; % Initialize best k assuming the first k tested will be odd and thus 1

    % Initialize index for storing accuracies in all_accuracies
    index = 1;
    
    % Loop through odd k values only
    for k = 1:2:max_k
        accuracy = 0;
        predicted_categories = cell(num_test, 1);

        % Classify test images
        for i = 1:num_test
            [~, idx] = mink(distances(i, :), k);
            neighbor_labels = train_labels(idx);
            [unique_labels, ~, label_indices] = unique(neighbor_labels);
            label_counts = accumarray(label_indices, 1);
            [~, max_count_idx] = max(label_counts);
            most_common_label = unique_labels{max_count_idx};

            % Assign predicted label
            predicted_categories{i} = most_common_label;

            % Check accuracy
            if strcmp(predicted_categories{i}, test_labels{i})
                accuracy = accuracy + 1;
            end
        end

        % Calculate accuracy
        accuracy = accuracy / num_test;
        
        % Store accuracy for current odd k
        all_accuracies(index) = accuracy;

        % Update best accuracy and best k
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_k = k;
        end

        % Increment index for the next odd k value
        index = index + 1;
    end

    % Create a table for k values and their corresponding accuracies
    k_accuracy_table = table(k_values, all_accuracies, 'VariableNames', {'k', 'Accuracy'});
    
    % Remove unused rows
    k_accuracy_table = k_accuracy_table(1:index-1, :);
    
    % Record best accuracy and corresponding k in the workspace
    best_k_accuracy = table(best_k, best_accuracy, 'VariableNames', {'Best_k', 'Best_Accuracy'});

end
