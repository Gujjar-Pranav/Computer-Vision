function [best_predicted_categories, best_model, bestTestAccuracy, bestOutOfBagError, best_num_trees] = random_forest_classify(train_image_feats, train_labels, test_image_feats, test_labels, num_trees_list, seed)
    %set seed  for reproducability of result
    rng(seed);
    
    % Initialize variables
    bestTestAccuracy = 0;
    best_num_trees = 0;
    best_predicted_categories = {};
    best_model = [];
    bestOutOfBagError = [];
    results = [];     
    for num_trees = num_trees_list
        % Train the Random Forest model using TreeBagger
        model = TreeBagger(num_trees, train_image_feats, train_labels, ...
            'Method', 'classification', 'OOBPrediction', 'On', ...
            'MinLeafSize', 2, 'NumPredictorsToSample', 'all', ...
            'MaxNumSplits', 500);

        % Predict the categories of the test data using the trained model
        [predicted_category_scores, ~] = predict(model, test_image_feats);
        predicted_categories = cellstr(predicted_category_scores);
        
        correct_predictions = sum(strcmp(predicted_categories, test_labels));
        currentTestAccuracy = correct_predictions / length(test_labels); 
        % Out of Bag error 
        outOfBagError = oobError(model);
        currentOutOfBagError = outOfBagError(end);
        results = [results; num_trees, currentTestAccuracy, currentOutOfBagError];

        if currentTestAccuracy > bestTestAccuracy
            bestTestAccuracy = currentTestAccuracy;
            best_num_trees = num_trees;
            best_model = model;
            best_predicted_categories = predicted_categories;
            bestOutOfBagError = currentOutOfBagError;
        end
    end
    r_table = array2table(results, 'VariableNames', {'NumTrees', 'TestAccuracy', 'OutOfBagError'});
    assignin('base', 'r_table', r_table);  
    fprintf('Best Number of Trees: %d with Test Accuracy: %.2f%% and OOB Error: %.4f\n', best_num_trees, bestTestAccuracy * 100, bestOutOfBagError);
end
