%% LEiDA Computation for Yeo7 Networks
% Computes leading eigenvectors from Hilbert-transformed data
% Processes data at Yeo network level (7 networks)
% Input: hilbert_data.mat containing hilbert_med and hilbert_con
% Output: eigenvectors_data_yeo7.mat containing network-level phase-based eigenvectors
%         with preserved subject-level structure

%% Setup paths
ROOT_DIR = fileparts(fileparts(fileparts(mfilename('fullpath'))));
PROCESSED_DIR = fullfile(ROOT_DIR, 'data', 'processed');
MAPPING_DIR = fullfile(ROOT_DIR, 'code', 'brain_roi_mappings');

% Add mapping directory to path
if ~exist(MAPPING_DIR, 'dir')
    error('Brain ROI mappings directory not found: %s', MAPPING_DIR);
end
addpath(MAPPING_DIR);

% Load Hilbert-transformed data
hilbert_file = fullfile(PROCESSED_DIR, 'hilbert_data.mat');
if ~exist(hilbert_file, 'file')
    error(['Hilbert transformed data not found.\n' ...
           'Please run run_hilbert_transform.m first.\n' ...
           'Expected file: %s'], hilbert_file);
end

fprintf('Loading Hilbert data from: %s\n', hilbert_file);
load(hilbert_file, 'hilbert_med', 'hilbert_con', 'subject_info');

% Verify we have the correct condition mapping
fprintf('Verifying condition mapping...\n');
fprintf('- hilbert_con: Controls during meditation (%s)\n', subject_info.condition_names{1});
fprintf('- hilbert_med: Expert meditators during meditation (%s)\n', subject_info.condition_names{2});

% Load and verify network definitions
networks = define_network_mappings();
% Yeo 7 networks (no SUB)
network_fields = {'VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN'};

% Verify network definitions
if ~all(isfield(networks, network_fields))
    error('Missing network definitions in define_network_mappings() for Yeo7');
end

% Initialize output
eigenvectors_data_yeo = struct();
groups = {'controls', 'meditators'};

% Process each group
for g = 1:length(groups)
    group = groups{g};
    fprintf('Computing Yeo7 network eigenvectors for %s...\n', group);
    
    % Select appropriate data
    if strcmp(group, 'meditators')
        data = hilbert_med;
    else
        data = hilbert_con;
    end
    
    % Initialize cell array to store subject-level eigenvectors
    eigenvectors_data_yeo.(group).subjects = cell(subject_info.num_subjects, 1);
    
    % Process each subject individually
    for subj = 1:subject_info.num_subjects
        fprintf('Processing %s subject %d/%d...\n', group, subj, subject_info.num_subjects);
        
        % Get subject-specific timepoints
        idx_start = subject_info.subject_indices{subj}(1);
        idx_end = subject_info.subject_indices{subj}(2);
        subject_data = data(:, idx_start:idx_end);
        
        % Average ROIs within each network for this subject
        subject_network_data = zeros(length(network_fields), size(subject_data, 2));
        for i = 1:length(network_fields)
            net = network_fields{i};
            rois = networks.(net).rois;
            subject_network_data(i, :) = mean(subject_data(rois, :), 1);
        end
        
        % Compute leading eigenvectors for this subject
        eigenvectors_data_yeo.(group).subjects{subj} = compute_leading_eigs(subject_network_data);
        
        % Validate dimensions - eigenvectors should be [timepoints × networks]
        [n_timepoints, n_networks] = size(eigenvectors_data_yeo.(group).subjects{subj});
        if n_timepoints ~= size(subject_data, 2) || n_networks ~= length(network_fields)
            error('Validation failed: Subject %d eigenvector dimensions (%d×%d) do not match expected (%d×%d)', ...
                  subj, n_timepoints, n_networks, size(subject_data, 2), length(network_fields));
        end
    end
    
    % For backward compatibility and validation, also create a concatenated version
    all_eigenvectors = [];
    for subj = 1:subject_info.num_subjects
        all_eigenvectors = [all_eigenvectors; eigenvectors_data_yeo.(group).subjects{subj}];
    end
    eigenvectors_data_yeo.(group).network = all_eigenvectors;
    
    % Add subject information
    eigenvectors_data_yeo.(group).n_subjects = subject_info.num_subjects;
    eigenvectors_data_yeo.(group).timepoints_per_subject = subject_info.timepoints_per_subject;
    eigenvectors_data_yeo.(group).subject_indices = subject_info.subject_indices;
    
    fprintf('Completed network eigenvector computation for %s\n', group);
end

% Add metadata
eigenvectors_data_yeo.info = subject_info;
eigenvectors_data_yeo.parameters.num_networks = length(network_fields);
eigenvectors_data_yeo.parameters.total_timepoints = size(hilbert_med, 2);
eigenvectors_data_yeo.network_info = networks;
eigenvectors_data_yeo.network_fields = network_fields;
eigenvectors_data_yeo.parameters.groups = groups;

% Save results to a Yeo7-specific file (create eigenvectors_data_yeo7 var before saving)
output_file = fullfile(PROCESSED_DIR, 'eigenvectors_data_yeo7.mat');
eigenvectors_data_yeo7 = eigenvectors_data_yeo; 
save(output_file, 'eigenvectors_data_yeo7');
fprintf('Network-level leading eigenvector computation for Yeo7 complete.\n');

% Print final dimensions
fprintf('\nFinal data dimensions:\n');
fprintf('Subject-level: Each subject has eigenvectors of size [%d timepoints × %d networks]\n', ...
    subject_info.timepoints_per_subject, length(network_fields));
fprintf('Group-level concatenated: [%d timepoints × %d networks]\n', ...
    size(eigenvectors_data_yeo.controls.network, 1), length(network_fields));
fprintf('Total subjects per group: %d\n', subject_info.num_subjects);

fprintf('Example eigenvector (subject 1, first timepoint):\n');
disp(eigenvectors_data_yeo.(group).subjects{1}(1, :));
