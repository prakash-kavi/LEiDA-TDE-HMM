%% Hilbert Transform for Meditation fMRI Data
% Applies preprocessing and Hilbert transform to raw fMRI data
% 1. Removes initial/final volumes (60 start, 20 end)
% 2. Bandpass filters (0.04-0.07 Hz)
% 3. Applies Hilbert transform
%
% Data columns in original file:
% Column 1 = Controls in resting state
% Column 2 = Controls meditating
% Column 3 = Meditators in resting state
% Column 4 = Meditators meditating
%
% This script processes only meditation conditions (columns 2 and 4)

clear all; close all; clc;

%% Parameters
TR = 2;
initial_volumes = 60;
final_volumes = 20;
num_subjects = 20;
num_rois = 116;
timepoints_per_subject = 440;
clean_timepoints = timepoints_per_subject - initial_volumes - final_volumes;  % 360

% Filter settings
fnq = 1/(2*TR);
flp = 0.04;
fhi = 0.07;
Wn = [flp/fnq fhi/fnq];
[bfilt,afilt] = butter(2, Wn);

%% Setup paths
ROOT_DIR = fileparts(fileparts(fileparts(mfilename('fullpath'))));
DATA_DIR = fullfile(ROOT_DIR, 'data', 'raw');
OUTPUT_DIR = fullfile(ROOT_DIR, 'data', 'processed');

% Create directories if needed
for dir = {DATA_DIR, OUTPUT_DIR}
    if ~exist(dir{1}, 'dir')
        mkdir(dir{1});
        warning('Created directory: %s', dir{1});
    end
end

%% Load and verify data
data_file = fullfile(DATA_DIR, 'tc_schaefer_med.mat');
if ~exist(data_file, 'file')
    error('Data file not found: %s', data_file);
end

%% Debug: Check original data structure
fprintf('\n==== DEBUG: Original Data Structure ====\n');
fprintf('Size of tc_schaefer_med: %s\n', mat2str(size(tc_schaefer_med)));
% Check first subject, first ROI, specific timepoints
test_subj = 2;  % Specifically check Subject 2 (the problematic one)
test_roi = 25;
test_time = 150;
fprintf('Subject %d, ROI %d, Time %d: Value = %.4f\n', ...
    test_subj, test_roi, test_time, tc_schaefer_med{test_subj, 2}(test_roi, test_time));
% Save this value to compare later
original_value = tc_schaefer_med{test_subj, 2}(test_roi, test_time);

fprintf('Loading data from: %s\n', data_file);
load(data_file, 'tc_schaefer_med');

if ~exist('tc_schaefer_med', 'var') || size(tc_schaefer_med, 2) ~= 4
    error('Data file must contain tc_schaefer_med with 4 columns');
end

%% Process meditation conditions only
% Column 2 = Controls meditating
% Column 4 = Meditators meditating
meditation_columns = [2, 4]; 
condition_names = {'med_con', 'med_med'};  % Controls meditation, Meditators meditation
hilbert_data = cell(1, 2);

for i = 1:2
    cond = meditation_columns(i);
    fprintf('Processing %s data (column %d)...\n', condition_names{i}, cond);
    
    % Initialize 3D array directly with proper dimensions
    cond_data = zeros(num_rois, timepoints_per_subject, num_subjects);
    
    % Explicitly copy each subject's data to the correct position
    for s = 1:num_subjects
        cond_data(:,:,s) = tc_schaefer_med{s, cond};
    end
    
    % Initialize output
    hilbert_cond = zeros(num_rois, clean_timepoints * num_subjects);
    
    for subj = 1:num_subjects
        fprintf('Processing %s subject %d/%d\n', condition_names{i}, subj, num_subjects);
        subject_data = cond_data(:,:,subj);
        
        % Process each ROI
        for roi = 1:num_rois
            signal = subject_data(roi,:);
            signal = signal - mean(signal);
            filtered = filtfilt(bfilt, afilt, signal);
            subject_data(roi,:) = hilbert(filtered);
        end
        
        % Remove initial/final volumes
        clean_data = subject_data(:, initial_volumes+1:end-final_volumes);
        
        % Store in output
        idx_start = ((subj-1) * clean_timepoints) + 1;
        idx_end = subj * clean_timepoints;
        hilbert_cond(:, idx_start:idx_end) = clean_data;
    end
    
    % Store in cell array
    hilbert_data{i} = hilbert_cond;
end

% Extract specific conditions
hilbert_con = hilbert_data{1};  % Controls meditating (column 2)
hilbert_med = hilbert_data{2};  % Meditators meditating (column 4)

%% Save metadata 
subject_info = struct();
subject_info.timepoints_per_subject = clean_timepoints;
subject_info.num_subjects = num_subjects;
subject_info.subject_indices = cell(num_subjects, 1);
subject_info.condition_names = condition_names;

% Store subject index for later reference
for subj = 1:num_subjects
    idx_start = ((subj-1) * clean_timepoints) + 1;
    idx_end = subj * clean_timepoints;
    subject_info.subject_indices{subj} = [idx_start, idx_end];
end

%% Save data
fprintf('Saving processed data...\n');

% Save meditation data
output_file = fullfile(OUTPUT_DIR, 'hilbert_data.mat');
save(output_file, 'hilbert_med', 'hilbert_con', 'subject_info');
fprintf('- Meditation data saved to: %s\n', output_file);

%% Print final dimensions
fprintf('\nFinal data dimensions:\n');
fprintf('Controls (meditating): %d ROIs × %d timepoints (%d timepoints per subject)\n', ...    
    size(hilbert_con, 1), size(hilbert_con, 2), clean_timepoints);
fprintf('Meditators (meditating): %d ROIs × %d timepoints (%d timepoints per subject)\n', ...    
    size(hilbert_med, 1), size(hilbert_med, 2), clean_timepoints);