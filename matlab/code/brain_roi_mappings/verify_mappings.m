function verify_mappings()
    % Load network mappings
    networks = define_network_mappings();
    
    % Get all ROIs
    all_rois = [];
    fields = fieldnames(networks);
    for i = 1:length(fields)
        all_rois = [all_rois, networks.(fields{i}).rois];
    end
    all_rois = unique(sort(all_rois));
    
    % Verify coverage
    fprintf('Network Coverage Analysis:\n');
    fprintf('Total unique ROIs: %d\n', length(all_rois));
    if length(all_rois) ~= 116
        warning('Not all 116 ROIs are covered by networks');
        missing = setdiff(1:116, all_rois);
        fprintf('Missing ROIs: %s\n', mat2str(missing));
    end
    
    % Check for overlaps
    fprintf('\nChecking for network overlaps...\n');
    for i = 1:length(fields)
        for j = i+1:length(fields)
            overlap = intersect(networks.(fields{i}).rois, ...
                              networks.(fields{j}).rois);
            if ~isempty(overlap)
                fprintf('Overlap between %s and %s: %s\n', ...
                        fields{i}, fields{j}, mat2str(overlap));
            end
        end
    end
end