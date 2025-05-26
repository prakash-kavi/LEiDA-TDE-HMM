# In your Python console or notebook
import pandas as pd
df = pd.read_csv('G:/leida_hmm/python/data/trained/tde_model_comparison.csv')
print(df[['group', 'n_states', 'test_free_energy', 'test_log_likelihood']])

# Verify relationship
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.scatter(df['test_free_energy'], df['test_log_likelihood'])
plt.xlabel('Free Energy')
plt.ylabel('Log-Likelihood')
plt.show()