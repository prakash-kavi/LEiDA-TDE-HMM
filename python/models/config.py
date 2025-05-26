# Define Yeo networks and attention-related networks
YEO_NETWORKS = [
    'VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN', 'SUB']

# Define networks of interest for Vipassana meditation
ATTENTION_NETWORKS = {
    'DAN': 'Dorsal Attention',       # Focus, sustained attention
    'VAN': 'Ventral Attention',      # Attention reorienting, meta-awareness
    'FPN': 'Frontoparietal',         # Executive control, regulation
    'DMN': 'Default Mode'            # Mind wandering, self-referential
}

# Map to Hasenkamp 4-state model
HASENKAMP_STATES = {
    'Focus': ['DAN', 'FPN'],                   # Sustained attention, executive control
    'Mind_Wandering': ['DMN'],                 # Default mode activation
    'Meta_Awareness': ['VAN', 'FPN'],          # Catching mind wandering
    'Redirect': ['VAN', 'DAN', 'FPN']          # Shifting attention back
}