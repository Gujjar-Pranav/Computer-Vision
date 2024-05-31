function [image_feats] = get_spatial_pyramids(image_paths, vocab, levels, mode, sift_type, step_size, bin_size)   
    vocab = single(vocab); 
    num_images = length(image_paths);
    num_words = size(vocab, 2);
    image_feats = zeros(num_images, num_words * sum(4.^(0:levels))); 
    weights = 2.^(-(levels:-1:0)); 

    for i = 1:num_images
        img = imread(image_paths{i});
        if strcmp(mode, 'grayscale') && size(img, 3) == 3
            img = rgb2gray(img); 
        end
        img = single(img); 
        idx_start = 1; 

        for l = 0:levels
            num_cells = 2^l;
            cell_size = floor(size(img, [1 2]) ./ num_cells);
            for x = 1:num_cells
                for y = 1:num_cells
                    xRange = (x-1)*cell_size(2)+1 : min(x*cell_size(2), size(img,2));
                    yRange = (y-1)*cell_size(1)+1 : min(y*cell_size(1), size(img,1));
                    cell_img = img(yRange, xRange, :); 
                    
                    % Handling DSIFT for both grayscale and color modes
                    if strcmp(sift_type, 'DSIFT')
                        if strcmp(mode, 'color') && size(cell_img, 3) == 3
                            % Process each color channel separately
                            histograms = zeros(num_words, 3);
                            for ch = 1:3
                                [~, cell_features] = vl_dsift(cell_img(:,:,ch), 'fast', 'step', step_size, 'size', bin_size);
                                cell_features = single(cell_features); 
                                if isempty(cell_features)
                                    histograms(:, ch) = zeros(num_words, 1);
                                else
                                    dists = vl_alldist2(vocab, cell_features);
                                    [~, min_indices] = min(dists);
                                    histograms(:, ch) = vl_binsum(zeros(num_words, 1), 1, min_indices);
                                end
                            end
                            histogram = sum(histograms, 2) / sum(sum(histograms));
                        else
                            % Grayscale or single channel processing
                            [~, cell_features] = vl_dsift(cell_img, 'fast', 'step', step_size, 'size', bin_size);
                            cell_features = single(cell_features); 
                            if isempty(cell_features)
                                histogram = zeros(num_words, 1);
                            else
                                dists = vl_alldist2(vocab, cell_features);
                                [~, min_indices] = min(dists);
                                histogram = vl_binsum(zeros(num_words, 1), 1, min_indices);
                                histogram = histogram / sum(histogram);  
                            end
                        end
                    elseif strcmp(sift_type, 'SIFT')
                        % Standard SIFT, assuming grayscale processing
                        if size(cell_img, 3) == 3
                            cell_img = rgb2gray(cell_img);
                        end
                        [~, d] = vl_sift(cell_img);
                        cell_features = single(d); 
                        if isempty(cell_features)
                            histogram = zeros(num_words, 1);
                        else
                            dists = vl_alldist2(vocab, cell_features);
                            [~, min_indices] = min(dists);
                            histogram = vl_binsum(zeros(num_words, 1), 1, min_indices);
                            histogram = histogram / sum(histogram);  
                        end
                    end

                    idx_end = idx_start + num_words - 1;
                    image_feats(i, idx_start:idx_end) = histogram' * weights(l+1);  
                    idx_start = idx_end + 1;
                end
            
        end
        image_feats(i, :) = image_feats(i, :) / norm(image_feats(i, :), 2); 
    end
end
