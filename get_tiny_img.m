function tiny_images = get_tiny_img(data_path, output_size, color_space, quantization_level, normalize)
    % Number of images 
    N = length(data_path);
    
    % Initialize the output matrix
    % For a 16x16 image, d would be 256 for grayscale, and 256*3 for color images (except CMYK which would be 256*4)
    if strcmp(color_space, 'grayscale')
        D = prod(output_size); % Grayscale images have only one channel
    elseif strcmp(color_space, 'cmyk')
        D = prod(output_size) * 4; % CMYK images have four channels
    else
        D = prod(output_size) * 3; % Other color images have three channels
    end
    
    tiny_images = zeros(N, D);
    
    for i = 1:N
        % Read image from given path
        img = imread(data_path{i});
        
        % Convert color space if needed
        switch lower(color_space)
            case 'hsv'
                img = rgb2hsv(img);
            case 'lab'
                img = rgb2lab(img);
            case 'grayscale'
                img = rgb2gray(img);
            case 'ycbcr'
                img = rgb2ycbcr(img);
            case 'cmyk'
                % Convert RGB to CMYK using below formula
                img_cmyk = 1 - double(img) / 255; % Convert to CMY
                k = min(img_cmyk, [], 3);
                img_cmyk = cat(3, img_cmyk, k); % Add K channel
                for j = 1:3 % Subtract K from CMY to get CMYK
                    img_cmyk(:,:,j) = img_cmyk(:,:,j) - k;
                end
                img = img_cmyk;
            case 'rgb'
               
            otherwise
                error('Unsupported color space.');
        end
        
        % Resize image
        img_resized = imresize(img, output_size);
        
        % Quantization
        if quantization_level > 0
            if ~strcmp(color_space, 'grayscale')
                img_resized = double(img_resized) / 255; % Scale to [0, 1] for color images
            end
            % Quantize each channel
            for c = 1:size(img_resized, 3)
                img_resized(:,:,c) = round(img_resized(:,:,c) * (quantization_level - 1)) / (quantization_level - 1);
            end
        end
        
        % Flatten image
        tiny_image = img_resized(:)';
        
        % Normalization
        if normalize
            tiny_image = double(tiny_image) - double(mean(tiny_image));
            tiny_image_norm = norm(tiny_image);
            if tiny_image_norm > 0
                tiny_image = tiny_image / tiny_image_norm;
            end
        end
        
        % Store in the output matrix
        tiny_images(i, :) = tiny_image;
    end
end
