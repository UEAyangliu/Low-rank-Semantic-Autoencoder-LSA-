function mappedL = mapLabel(L, classes)
    mappedL = -1 * ones(size(L));
    for i=1:length(classes)
        mappedL(L == i) = classes(i);
    end

    
