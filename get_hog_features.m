function image_feats = get_hog_features(data_path, mode, seed)
    rng(seed);
    % Set up Parameters for various experiments
    cell_size = 64; % spacify the cell size
    block_size = 2; % specify the block size
    num_bins = 83; % specify the number of bins 
    img_size = [256, 256];  % spacify the image size

    % Calculate dimensions for feature vector
    % Calculate dimensions for cells in x,y directions
    num_cells_x = floor(img_size(2) / cell_size);
    num_cells_y = floor(img_size(1) / cell_size);
    % calculate dimensions for blocks in x,y direction
    num_blocks_x = num_cells_x - block_size + 1;
    num_blocks_y = num_cells_y - block_size + 1;
    % calculate histogram per block
    hist_per_block = block_size^2 * num_bins;
    % findout feature length
    feature_length = hist_per_block * num_blocks_x * num_blocks_y; 

    num_images = length(data_path);
    image_feats = zeros(num_images, feature_length);
    
    for i = 1:num_images
        img = imread(data_path{i});
        
        % Handle color mode
        if strcmp(mode, 'color') && size(img, 3) == 3
            hog_features = [];
            for channel = 1:size(img, 3)
                channel_img = img(:, :, channel);
                channel_hog = extractHOGFeatures(channel_img, ...
                    'CellSize', [cell_size, cell_size], ...
                    'BlockSize', [block_size, block_size], ...
                    'NumBins', num_bins);
                hog_features = [hog_features, channel_hog];
            end
        % Handle grayscale mode
        elseif strcmp(mode, 'grayscale')
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            hog_features = extractHOGFeatures(img, ...
                'CellSize', [cell_size, cell_size], ...
                'BlockSize', [block_size, block_size], ...
                'NumBins', num_bins);
        else
            error('Invalid mode. Mode must be "grayscale" or "color".');
        end
        
        % L2 normalization (Euclidean normalization)
        hog_features = hog_features / norm(hog_features);
        
        % Ensure feature vector length matches the desired length
        if length(hog_features) < feature_length
            hog_features = [hog_features, zeros(1, feature_length - length(hog_features))];
        else
            hog_features = hog_features(1:feature_length);
        end
        
        image_feats(i, :) = hog_features;
    end
end
