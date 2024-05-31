% Michal Mackiewicz, UEA 
% This code has been adapted from the code 
% prepared by James Hays, Brown University

%% Step 0: Set up parameters, vlfeat, category list, and image paths.

%FEATURE = 'tiny image';
%FEATURE = 'colour histogram';
%FEATURE = 'bag of sift';
FEATURE = 'spatial pyramids';
%FEATURE = 'histogram of gradient';


%CLASSIFIER = 'nearest neighbor';
CLASSIFIER = 'support vector machine';
%CLASSIFIER = 'random forest';

% Set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
%run('vlfeat/toolbox/vl_setup')

data_path = 'D:\Pranav -UK\Msc Data Science-UEA-Norwich-UK\Computer vision\CW2\extra_code_CW2\data';

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
   
%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    
%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 100; 

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell      
%   test_image_paths   1500x1   cell           
%   train_labels       1500x1   cell         
%   test_labels        1500x1   cell          

%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the 
% dimensionality of each image representation. See the starter code for
% each function for more details.

fprintf('Using %s representation for images\n', FEATURE)
% Specify vocabulary size for Build vocabulary
vocab_size = 500; % Change as needed for experiments for bag of sift feature only
stored_vocab_size = vocab_size; % Define stored_vocab_size
mode = 'color';  % options 'grayscale' or 'color' for various experiments.here color means consider 'RGB'
sift_type = 'DSIFT';  % Options  'SIFT' or 'DSIFT' for variours experiments.
levels = 2; % spatial pyramid level [0,1,2] for Dense Sift.
step_size = 4; % spatial Pyramid step size [4,8,12,16] for Dense Sift.
bin_size = 4; % spatial bins used for feature extraction for Dense Sift.
%seed =123; % enable for HOG  Feature and Random forest classifer 


switch lower(FEATURE)    
    case 'tiny image'
        %You need to reimplement get_tiny_images. Allow function to take
        %parameters e.g. feature size.
        
        % image_paths is an N x 1 cell array of strings where each string is an
        %  image path on the file system.
        % image_feats is an N x d matrix of resized and then vectorized tiny
        %  images. E.g. if the images are resized to 16x16, d would equal 256.
        
        % To build a tiny image feature, simply resize the original image to a very
        % small square resolution, e.g. 16x16. You can either resize the images to
        % square while ignoring their aspect ratio or you can crop the center
        % square portion out of each image. Making the tiny images zero mean and
        % unit length (normalizing them) will increase performance modestly.
        
        train_image_feats = get_tiny_images(train_image_paths);
        test_image_feats  = get_tiny_images(test_image_paths);
    case 'colour histogram'
        %You should allow get_colour_histograms to take parameters e.g.
        %quantisation, colour space etc.
        train_image_feats = get_colour_histograms(train_image_paths);
        test_image_feats  = get_colour_histograms(test_image_paths);

     case 'bag of sift'
        % Initialize the recomputation features if require
        recompute_features = true;
    
        % Check for the existence of vocab.mat and image_feats.mat files
        if exist('g_V500_bs_svm_won_vocab.mat', 'file')
            load('g_V500_bs_svm_won_vocab.mat', 'vocab', 'stored_vocab_size');  
            disp(['Stored vocabulary size: ', num2str(stored_vocab_size)]);
            disp(['Desired vocabulary size: ', num2str(vocab_size)]);
            if stored_vocab_size ~= vocab_size  % Check if the current vocab size matches the desired size
                fprintf('Vocabulary size has changed. Recomputing vocabulary and features...\n');
                %buid vocabulary 
                vocab = build_vocabulary1(train_image_paths, vocab_size, mode);
                % Update stored vocabulary size
                stored_vocab_size = vocab_size;
                save('g_V500_bs_svm_won_vocab.mat', 'vocab', 'stored_vocab_size', '-v7.3');
            else
                if exist('g_V500_bs_svm_won_image_feats.mat', 'file')
                    fprintf('Loading existing image features...\n');
                    load('g_V500_bs_svm_won_image_feats.mat', 'train_image_feats', 'test_image_feats');
                    recompute_features = false;  
                end
            end
        else
            fprintf('No existing dictionary found. Computing one from training images\n');
            %Build Vocabulary
            vocab = build_vocabulary1(train_image_paths, vocab_size, mode);            
            stored_vocab_size = vocab_size;
            save('g_V500_bs_svm_won_vocab.mat', 'vocab', 'stored_vocab_size', '-v7.3');
        end

        if recompute_features
            fprintf('Computing bags of SIFT features...\n');
            train_image_feats = get_bags_of_sifts1(train_image_paths, vocab, mode, sift_type,step_size,bin_size);
            test_image_feats = get_bags_of_sifts1(test_image_paths, vocab, mode, sift_type,step_size,bin_size);
            save('g_V500_bs_svm_won_image_feats.mat', 'train_image_feats', 'test_image_feats', '-v7.3');
        end   
     
      case 'spatial pyramids'
          % Initialize the recomputation feature if require
        recompute_features = true;
    
        % Check for the existence of vocab.mat and image_feats.mat files
        if exist('c_V500_sp_svm_vocab.mat', 'file')
            load('c_V500_sp_svm_vocab.mat', 'vocab', 'stored_vocab_size');  
            disp(['Stored vocabulary size: ', num2str(stored_vocab_size)]);
            disp(['Desired vocabulary size: ', num2str(vocab_size)]);
            if stored_vocab_size ~= vocab_size  % Check if the current vocab size matches the desired size
                fprintf('Vocabulary size has changed. Recomputing vocabulary and features...\n');
                %Build Vocabulary
                vocab = build_vocabulary1(train_image_paths, vocab_size, mode);
                stored_vocab_size = vocab_size;
                save('c_V500_sp_svm_vocab.mat', 'vocab', 'stored_vocab_size', '-v7.3');
            else
                if exist('c_V500_sp_svm_image_feats.mat', 'file')
                    fprintf('Loading existing image features...\n');
                    load('c_V500_sp_svm_image_feats.mat', 'train_image_feats', 'test_image_feats');
                    recompute_features = false;  
                end
            end
        else
            fprintf('No existing dictionary found. Computing one from training images\n');
            % Build Vocabulary
            vocab = build_vocabulary1(train_image_paths, vocab_size, mode);
            save('c_V500_sp_svm_vocab.mat', 'vocab', 'stored_vocab_size', '-v7.3');
        end

        % Recompute image features if necessary
        if recompute_features
            fprintf('Computing spatial pyramids features...\n');
            train_image_feats = get_spatial_pyramids(train_image_paths, vocab,levels, mode,sift_type,step_size,bin_size);
            test_image_feats = get_spatial_pyramids(test_image_paths, vocab,levels, mode,sift_type,step_size,bin_size);
            save('c_V500_sp_svm_image_feats.mat', 'train_image_feats', 'test_image_feats', '-v7.3');
        end

    case'histogram of gradient'
        train_image_feats = get_hog_features(train_image_paths,mode,seed);
        test_image_feats  = get_hog_features(test_image_paths,mode,seed);

end
%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)

switch lower(CLASSIFIER)    
    case 'nearest neighbor'
    %Here, you need to reimplement nearest_neighbor_classify. My P-code
    %implementation has k=1 set. You need to allow for varying this
    %parameter.
    max_k = 33;
    %This function will predict the category for every test image by finding
    %the training image with most similar features. Instead of 1 nearest
    %neighbor, you can vote based on k nearest neighbors which will increase
    %performance (although you need to pick a reasonable value for k).
    
    % image_feats is an N x d matrix, where d is the dimensionality of the
    %  feature representation.
    % train_labels is an N x 1 cell array, where each entry is a string
    %  indicating the ground truth category for each training image.
    % test_image_feats is an M x d matrix, where d is the dimensionality of the
    %  feature representation. You can assume M = N unless you've modified the
    %  starter code.
    % predicted_categories is an M x 1 cell array, where each entry is a string
    %  indicating the predicted category for each test image.
    % Useful functions: pdist2 (Matlab) and vl_alldist2 (from vlFeat toolbox)
    % Here, you call the nearest_neighbor_classify function with only the necessary output arguments
    [predicted_categories, best_k, best_accuracy, k_accuracy_table] = nearest_neighbor_classify(train_image_feats, ...
        train_labels, test_image_feats, test_labels, max_k);

    % Use the obtained best_k for classification
    [predicted_categories, best_k, best_accuracy] = nearest_neighbor_classify(train_image_feats, ...
        train_labels, test_image_feats, test_labels, best_k);
         %predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats);
    case 'support vector machine'
    % Call the support vectore machine function with only the necessary output arguments    
    [best_lambda, best_accuracy, predicted_categories] = svm_classify1(train_image_feats, train_labels, test_image_feats, test_labels);
    
    case 'random forest'
    % Number of Tree list      
    num_trees_list = [50, 100, 150, 200, 250, 300, 350];
    % Call the random forest function with only the necessary output arguments
    [best_predicted_categories, best_model, bestTestAccuracy, bestOutOfBagError, best_num_trees] = random_forest_classify(train_image_feats, train_labels, test_image_feats, test_labels, num_trees_list, seed);
    % use predicated categories for confusion matrix in step 3
    predicted_categories = best_predicted_categories;

end

%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section. 

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?
create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)
%%
%clear