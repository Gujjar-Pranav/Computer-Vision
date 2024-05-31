% last edit Feb 2022, Michal Mackiewicz, UEA
% This code has been adapted from the code 
% prepared by James Hays, Brown University

% For coursework 1 you need to implement 'tiny_image' and 'colour histogram' in "Step
% 1" and 'nearest neighbor' in Step 2

%% Step 0: Set up parameters, vlfeat, category list, and image paths.
% Define all parameters in a structured array
% Parameters for tiny images
tiny_img_params = struct(...
    'output_size', [4, 4], ...% Output size:[2 2],[4 4],[8 8],[16 16],[32 32],[64 64] [128 128],[256 256]
    'color_space', 'rgb', ... % Color space: 'rgb', 'hsv', 'lab', 'ycbcr','grayscale','cmyk'
    'quantization_level', 0 ...% Quantization level: 0,4,8,16,32
    , ...
    'normalize', true); % Normalization: Ture

% Define parameters for color histograms
color_hist_params = struct(...
    'color_space', 'hsv', ... % Color space: 'rgb', 'hsv', 'lab', 'ycbcr','grayscale','cmyk'
    'num_bins', 8, ... % Number of bins per channel for the histogram : 4,8,16,32,64,128,256,512
    'normalize', true); ... % Whether to normalize the histograms)




%You need to experiment  with three
%combinations of features / classifiers:
% 1) Tiny image features and nearest neighbor classifier
% 2) Colour histogram and nearest neighbor classifier
%The starter code is initialized to 'tiny image' features and 'nearest
%neighbor' classifier and should run as is using my obfuscated pcode
%implementation. You need to reimplement the above two functions and also
%implement the 'colour histogram' feature extraction.

%FEATURE = 'tiny image'; % Its default  P code file
%FEATURE = 'tiny img'   % Enable this code which is written by us. also
%enable  line number 175 to 179. Input parameters will display at command
%line
FEATURE = 'color histogram with parameters'; % Enable this code which is written by us.
% also enable line no 183 to 186.Input parameters will display at command
%line


CLASSIFIER = 'nearest neighbor';

% I suggest you install and setup VLFeat toolbox. You may not need it in
% coursework 1, but you will definitely need it in coursework 2.
% Set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
%run('vlfeat/toolbox/vl_setup')


data_path = 'D:\Pranav -UK\Msc Data Science-UEA-Norwich-UK\Computer vision\CW1\data';

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

    % Our Own write up tiny image Feature extraction function for CW_1.       
    case 'tiny img'
        % tiny img
        % Pass tiny_img_params directly to the function

        train_image_feats = get_tiny_img(train_image_paths, ...
            tiny_img_params.output_size, tiny_img_params.color_space, ...
            tiny_img_params.quantization_level, tiny_img_params.normalize);
        test_image_feats = get_tiny_img(test_image_paths, ...
            tiny_img_params.output_size, tiny_img_params.color_space, ...
            tiny_img_params.quantization_level, tiny_img_params.normalize);
    
    case 'color histogram with parameters'

    % pass color histogram parameters directly to the function
        train_image_feats = get_color_histograms_color_space(train_image_paths, ...
            color_hist_params.color_space, ...
            color_hist_params.num_bins, ...
            color_hist_params.normalize);
        test_image_feats = get_color_histograms_color_space(test_image_paths, ...
            color_hist_params.color_space, ...
            color_hist_params.num_bins, ...
            color_hist_params.normalize);    
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
    max_k = 21;
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
    %predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,test_labels);
end
% Comment out following to Display input parameters for Tiny img in command line
%disp(['Feature Extraction: Tiny img']);
%disp(['Output Size: ', mat2str(tiny_img_params.output_size)]);
%disp(['Color Space: ', num2str(tiny_img_params.color_space)]);
%disp(['Quantization_level: ', mat2str(tiny_img_params.quantization_level)]);
%disp(['Normalize: ',num2str(tiny_img_params.normalize)])


% Comment out following to Display input parameters for Color Histogram color space in command line.
disp(['Feature Extraction: color histogram with parameters']);
disp(['Color Space: ', color_hist_params.color_space]);
disp(['Number of Bins: ', num2str(color_hist_params.num_bins)]);
disp(['Normalize: ', mat2str(color_hist_params.normalize)]);
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


