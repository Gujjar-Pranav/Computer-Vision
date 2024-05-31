function image_feats = get_bags_of_sifts1(train_image_paths, vocab, mode, sift_type, step_size,bin_size)   
    % Get the number of images and vocabulary size and initialize matrix for image feature
    num_images = length(train_image_paths);
    vocab_size = size(vocab, 2);
    image_feats = zeros(num_images, vocab_size);
    for i = 1:num_images
        img = imread(train_image_paths{i});
        % Process images based on mode ('color' or 'grayscale') and Extract
        % sift descriptor for each color channel
        if strcmp(mode, 'color') && size(img, 3) == 3
            img = single(img);
            [~, descR] = vl_sift(img(:,:,1));
            [~, descG] = vl_sift(img(:,:,2));
            [~, descB] = vl_sift(img(:,:,3));
            descriptors = [descR, descG, descB];
        elseif size(img, 3) == 3
            img = single(rgb2gray(img));
            if strcmp(sift_type, 'DSIFT')
                [~, descriptors] = vl_dsift(img, 'fast', 'Step', step_size, 'Size', bin_size);
            else
                [~, descriptors] = vl_sift(img);
            end
        else
            img = single(img);
            if strcmp(sift_type, 'DSIFT')
                [~, descriptors] = vl_dsift(img, 'fast', 'Step', step_size, 'Size', bin_size);
            else
                [~, descriptors] = vl_sift(img);
            end
        end
        descriptors = single(descriptors);
        % Compute distances to visual words 
        D = vl_alldist2(descriptors, vocab);
        [~, min_indexes] = min(D, [], 2);
        % Generate histogram of visual word occurances
        image_feats(i, :) = histcounts(min_indexes, 1:(vocab_size+1));
    end
    return;
end
