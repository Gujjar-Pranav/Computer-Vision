function vocab = build_vocabulary1(train_image_paths, vocab_size, mode)   
    % Initialize empty matrix to store descriptors 
    descriptors = [];
    for i = 1:length(train_image_paths)
        img = imread(train_image_paths{i});    
        % Process image based on mode ('color' and 'grayscale') 
        if strcmp(mode, 'color') && size(img, 3) == 3    
            % convert image into single precision and extract sift
            % descriptor for each color channel
            img = single(img); 
            [~, descR] = vl_sift(img(:,:,1)); 
            [~, descG] = vl_sift(img(:,:,2));  
            [~, descB] = vl_sift(img(:,:,3)); 
            descriptors = [descriptors, descR, descG, descB];
        elseif size(img, 3) == 3
            img = single(rgb2gray(img));  
            [~, desc] = vl_sift(img);
            descriptors = [descriptors, desc];
        else
            img = single(img);  
            [~, desc] = vl_sift(img);
            descriptors = [descriptors, desc];
        end
    end   
    descriptors = single(descriptors);
    % Perform k-means clustering to build vocabulary
    [vocab, ~] = vl_kmeans(descriptors, vocab_size, 'algorithm', 'Elkan', 'NumRepetitions', 10);
end
