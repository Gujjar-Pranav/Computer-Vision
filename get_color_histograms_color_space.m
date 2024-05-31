function [histograms, distance_matrix] = get_color_histograms_color_space(data_path, color_space, num_bins, normalize)
    
    % Check if normalization is requested
    if normalize
        normalize_flag = 'probability'; % Normalize to sum up to 1
    else
        normalize_flag = 'count'; % No normalization
    end
    
    % Number of images
    N = length(data_path);
    
    % Determine the size of the histogram vector for each image based on color space
    switch lower(color_space)
        case {'rgb', 'hsv', 'lab', 'ycbcr'}
            d = 3 * num_bins;
        case 'grayscale'
            d = num_bins;
        case 'cmyk'
            d = 4 * num_bins;
        otherwise
            error('Unsupported color space.');
    end
    
    histograms = zeros(N, d);
    
    for i = 1:N
        % Read image
        img = imread(data_path{i});
        
        % Convert color space if needed
        switch lower(color_space)
            case 'rgb'
                % Handle RGB color space
            case 'hsv'
                % Convert to HSV and handle
                img = rgb2hsv(img);
            case 'lab'
                % Convert to LAB and handle
                img = rgb2lab(img);
            case 'ycbcr'
                % Convert to YCbCr and handle
                img = rgb2ycbcr(img);
            case 'grayscale'
                % Convert to grayscale and handle
                img = rgb2gray(img);
            case 'cmyk'
                % Convert to CMYK and handle
                img = rgb2cmyk_custom(img);
            otherwise
                error('Unsupported color space.');
        end
        
        % Compute histogram for each channel
        if strcmp(color_space, 'grayscale')
            histograms(i, :) = histcounts(img(:), num_bins, 'Normalization', normalize_flag);
        elseif strcmp(color_space, 'cmyk')
            for c = 1:4
                channel_values = img(:,:,c);
                histograms(i, (c-1)*num_bins+1:c*num_bins) = histcounts(channel_values(:), num_bins, 'Normalization', normalize_flag);
            end
        else
            % For other color spaces (RGB, HSV, LAB, YCbCr)
            for c = 1:3
                channel_values = img(:,:,c);
                % For YCbCr and LAB, scale values to [0, 1] for consistent histogram computation
                if strcmp(color_space, 'ycbcr') || strcmp(color_space, 'lab')
                    channel_values = double(channel_values) / 255;
                end
                histograms(i, (c-1)*num_bins+1:c*num_bins) = histcounts(channel_values(:), num_bins, 'Normalization', normalize_flag);
            end
        end
    end
    
    % Compute pairwise distances between histograms using Euclidean distance
    distance_matrix = pdist2(histograms, histograms);
    
    % Record the histograms and distance matrix in the workspace
    assignin('base', 'computed_histograms', histograms);
    assignin('base', 'computed_distance_matrix', distance_matrix);
end

function cmyk_img = rgb2cmyk_custom(rgb_img)
    % RGB to CMYK conversion
    % Normalize RGB values to [0, 1]
    rgb_img = double(rgb_img) / 255;
    
    % Separate RGB channels
    R = rgb_img(:,:,1);
    G = rgb_img(:,:,2);
    B = rgb_img(:,:,3);
    
    % Compute CMY components
    C = 1 - R;
    M = 1 - G;
    Y = 1 - B;
    
    % Compute K (black) channel
    K = min(min(C, M), Y);
    
    % Avoid division by zero
    C(K == 1) = 0;
    M(K == 1) = 0;
    Y(K == 1) = 0;
    
    % CMYK image
    cmyk_img = cat(3, C, M, Y, K);
end
