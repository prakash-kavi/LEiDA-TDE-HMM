function eigen_out = compute_leading_eigs(input_data)
    % Compute the leading eigenvector for each time point from phase data.
    % 
    % INPUT:
    %   input_data: Matrix [N_areas, T] where each column contains complex values 
    %               from Hilbert transform, representing phase and amplitude information.
    %
    % OUTPUT:
    %   eigen_out: Matrix [T, N_areas] where each row contains the leading eigenvector
    %              of the phase synchronization matrix at that timepoint, with consistent 
    %              sign convention applied.
    %
    % Note: This extracts phase synchronization patterns between brain regions
    % using only the phase component (angle) of the Hilbert-transformed data.
    
    [N_areas, T] = size(input_data);
    eigen_out = zeros(T, N_areas);
    
    for t = 1:T
        % Extract phase information at time point t (ignoring amplitude)
        phases = angle(input_data(:, t));

        % Create phase synchronization matrix (instantaneous functional connectivity)
        % Values close to 1 indicate in-phase regions, values close to -1 indicate
        % anti-phase regions, and values near 0 indicate uncorrelated phases
        iFC = cos(bsxfun(@minus, phases, phases'));
        
        % Extract the leading eigenvector (associated with largest eigenvalue)
        % This represents the dominant pattern of phase relationships at this timepoint
        [V1, ~] = eigs(iFC, 1);
        
        % Apply sign convention for consistency:
        % This ensures the same brain state is represented by the same eigenvector
        % orientation across timepoints, as eigenvectors can be arbitrarily flipped
        % (v and -v are both valid eigenvectors for the same eigenvalue)
        if mean(V1 > 0) > 0.5
            V1 = -V1;
        elseif mean(V1 > 0) == 0.5 && sum(V1(V1 > 0)) > -sum(V1(V1 < 0))
            V1 = -V1;
        end
        
        eigen_out(t, :) = V1;
    end
end