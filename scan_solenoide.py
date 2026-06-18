"""
Script to scan solenoid field values and extract final beam size.
Runs SEMrz.py with varying por, por2 and por3 parameters and plots B vs beam size.
"""
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

def plot_solenoid_scan(results_df):
    """Plot solenoid field vs final beam size"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract unique species
    species_list = results_df['species'].unique()
    
    for idx, sp in enumerate(species_list):
        if idx >= 4:
            break
        ax = axes.flat[idx]
        sp_data = results_df[results_df['species'] == sp]
        
        if sp_data.empty:
            continue
        
        # Scatter plot for each (por, por2, por3) combination
        scatter = ax.scatter(sp_data['B_total'], sp_data['sigma_r']*1e3,
                           s=100, c=sp_data['por'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('Total B field (por + por2 + por3)')
        ax.set_ylabel('Final sigma_r (mm)')
        ax.set_title(f'Beam size vs Field for {sp}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='por')
    
    plt.tight_layout()
    plt.savefig('solenoid_scan_results.png', dpi=150)
    print("Scan plot saved to solenoid_scan_results.png")
    plt.close()

# Scan parameters
por_values = np.linspace(0.0, 2.0, 41)  # Solenoid 1 field scale
por2_values = np.linspace(0.0, 2.0, 41)  # Solenoid 2 field scale
por3_values = np.linspace(0.045, 0.065, 21)  # Solenoid 3 field scale
por_values = [0.1329]
por2_values = [0.0991]
# Output file for scan results
scan_results_file = 'solenoid_scan_results.csv'

# Clear previous results
if os.path.exists(scan_results_file):
    os.remove(scan_results_file)
    #remove all cgm files to avoid accumulation
    os.remove("*.cgm*")

# Initialize results list
results = []

print("Starting solenoid field scan...")
print(f"Por values: {por_values}")
print(f"Por2 values: {por2_values}")
print(f"Por3 values: {por3_values}")

# Scan loop
total_runs = len(por_values) * len(por2_values) * len(por3_values)
run_number = 0
for i, por in enumerate(por_values):
    for j, por2 in enumerate(por2_values):
        for k, por3 in enumerate(por3_values):
            run_number += 1
            print(
                f"\n[{run_number}/{total_runs}] Running SEMrz with "
                f"por={por:.4f}, por2={por2:.4f}, por3={por3:.4f}"
            )

            # Clean up previous beam_size_history.csv if exists
            if os.path.exists('beam_size_history.csv'):
                os.remove('beam_size_history.csv')

            # Clean up previous .cgm files to avoid accumulation
            for cgm_file in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.cgm')):
                os.remove(cgm_file)

            # Run SEMrz.py with current parameters
            try:
                result = subprocess.run(
                    ['python', 'SEMrz.py', str(por), str(por2), str(por3)],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=True,
                    timeout=300,  # 5 minute timeout
                    text=True
                )

                if result.returncode != 0:
                    print(f"Error running SEMrz.py: {result.stderr}")
                    continue
            except subprocess.TimeoutExpired:
                print(f"Timeout for por={por:.4f}, por2={por2:.4f}, por3={por3:.4f}")
                continue
            except Exception as e:
                print(f"Exception: {e}")
                continue

            # Read beam_size_history.csv to get final values
            if not os.path.exists('beam_size_history.csv'):
                print(f"No beam_size_history.csv found for por={por:.4f}, por2={por2:.4f}, por3={por3:.4f}")
                continue

            try:
                data = pd.read_csv('beam_size_history.csv')
                if data.empty:
                    print(f"beam_size_history.csv is empty for por={por:.4f}, por2={por2:.4f}, por3={por3:.4f}")
                    continue

                # Get the last row (final values)
                last_row = data.iloc[-1]
                final_iter = last_row['iter']
                species = last_row['species']
                n_particles = last_row['n_particles']
                sigma_x_final = last_row['sigma_x']
                sigma_y_final = last_row['sigma_y']
                sigma_r_final = last_row['sigma_r']

                results.append({
                    'por': por,
                    'por2': por2,
                    'por3': por3,
                    'B_total': por + por2 + por3,  # Combined field
                    'final_iter': final_iter,
                    'species': species,
                    'n_particles': n_particles,
                    'sigma_x': sigma_x_final,
                    'sigma_y': sigma_y_final,
                    'sigma_r': sigma_r_final
                })

                print(f"Final sigma_r: {sigma_r_final:.6e} m")

            except Exception as e:
                print(f"Error reading beam_size_history.csv: {e}")
                continue

# Save results to CSV
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(scan_results_file, index=False)
    print(f"\nScan results saved to {scan_results_file}")
    
    # Plot B vs beam size
    plot_solenoid_scan(results_df)
else:
    print("No results collected!")

print("Solenoid scan completed!")
