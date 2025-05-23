import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from ../constants import FileNames

# PATH_TO_DATA = FileNames.RECORDING_FRAMES
# PATH_TO_DATA =  "C:\Users\maaya\Desktop\University\year3\labPrut\RecordingFrames.csv"
PATH_TO_DATA = "RecordingFrames.csv"

def process_monkey_trials(file_path):
    # Read the data (auto-detect separator, and handle potential issues)
    df = pd.read_csv(file_path, sep=None, engine='python')
    
    # Clean up column names (strip whitespaces)
    df.columns = df.columns.str.strip()
    
    # Fix potential column naming issues (e.g., lowercase 'type')
    if 'type' in df.columns:
        df.rename(columns={'type': 'Type'}, inplace=True)
    
    # Check if 'Type' column exists
    if 'Type' not in df.columns:
        raise KeyError("'Type' column not found in the data. Please check the file structure.")
    
    # Function to calculate solve time for boxes
    def calculate_box_solve_times(row):
        solve_times = {}
        
        for i in range(1, 5):  # Assuming max 4 boxes
            number_box_col = f'number_box_{i}'
            go_col = f'Go{i}'
            stop_col = f'Stop{i}'
            
            try:
                # Check if all required columns exist
                if all(col in row.index for col in [number_box_col, go_col, stop_col]):
                    # Check if all values are valid numbers
                    if all(pd.notna(row[col]) for col in [number_box_col, go_col, stop_col]):
                        box_num = int(row[number_box_col])
                        
                        # Validate box number is in expected range
                        if not 1 <= box_num <= 4:
                            print(f"Warning: Invalid box number {box_num} found in data")
                            continue
                            
                        # Calculate solve time
                        solve_time = row[stop_col] - row[go_col]
                        
                        # Validate solve time is positive
                        if solve_time < 0:
                            print(f"Warning: Negative solve time found for box {box_num}")
                            continue
                            
                        solve_times[f'box_{box_num}_solve_time'] = solve_time
                        
            except ValueError as e:
                print(f"Error processing box {i}: {e}")
                continue
                
        return pd.Series(solve_times)
    
    # Apply box solve time calculation
    box_solve_times = df.apply(calculate_box_solve_times, axis=1)
    
    # Combine original dataframe with calculated solve times
    df_processed = pd.concat([df, box_solve_times], axis=1)
    
    # Separate HFS and Control trials
    hfs_trials = df_processed[df_processed['Type'] == 'HFS']
    control_trials = df_processed[df_processed['Type'] == 'control']
    
    # Dynamically find box solve time columns
    box_solve_cols = [col for col in df_processed.columns if 'box_' in col and '_solve_time' in col]
    
    # Calculate average and median solve times
    hfs_avg_solve_times = hfs_trials[box_solve_cols].mean()
    hfs_median_solve_times = hfs_trials[box_solve_cols].median()
    control_avg_solve_times = control_trials[box_solve_cols].mean()
    control_median_solve_times = control_trials[box_solve_cols].median()
    
    # Statistical summaries
    hfs_solve_times_summary = hfs_trials[box_solve_cols].describe()
    control_solve_times_summary = control_trials[box_solve_cols].describe()
    
    return {
        'full_dataframe': df_processed,
        'hfs_trials': hfs_trials,
        'control_trials': control_trials,
        'hfs_avg_solve_times': hfs_avg_solve_times,
        'hfs_median_solve_times': hfs_median_solve_times,
        'control_avg_solve_times': control_avg_solve_times,
        'control_median_solve_times': control_median_solve_times,
        'hfs_solve_times_summary': hfs_solve_times_summary,
        'control_solve_times_summary': control_solve_times_summary
    }


# Example of reading data
def readData(filePath: str):
    df = pd.read_csv(filePath, sep=None, engine='python')
    print("Column names:", df.columns)
    print(df)

def plot_solve_times_comparison(results):
    # Get the average and median solve times
    hfs_avg = results['hfs_avg_solve_times']
    control_avg = results['control_avg_solve_times']
    hfs_median = results['hfs_median_solve_times']
    control_median = results['control_median_solve_times']
    
    # Get standard deviations
    hfs_std = results['hfs_trials'][[f'box_{i}_solve_time' for i in range(1, 5)]].std()
    control_std = results['control_trials'][[f'box_{i}_solve_time' for i in range(1, 5)]].std()
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Set the positions for the bars
    x = np.arange(4)  # 4 boxes
    width = 0.35  # width of the bars
    
    # Create bars for averages with error bars
    plt.bar(x - width/2, 
            [control_avg[f'box_{i}_solve_time'] for i in range(1, 5)],
            width, 
            label='Control Avg',
            color='royalblue',
            # yerr=[control_std[f'box_{i}_solve_time'] for i in range(1, 5)], // std deviation 
            capsize=5)
    
    plt.bar(x + width/2, 
            [hfs_avg[f'box_{i}_solve_time'] for i in range(1, 5)],
            width, 
            label='HFS Avg',
            color='lightcoral',
            # yerr=[hfs_std[f'box_{i}_solve_time'] for i in range(1, 5)], // std deviation 
            capsize=5)
    
    # Plot median values as scatter points
    plt.scatter(x - width/2, 
                [control_median[f'box_{i}_solve_time'] for i in range(1, 5)], 
                color='darkblue', label='Control Median', zorder=5, s=100, marker='o')
    plt.scatter(x + width/2, 
                [hfs_median[f'box_{i}_solve_time'] for i in range(1, 5)], 
                color='darkred', label='HFS Median', zorder=5, s=100, marker='o')
    
    # Add median values beneath the dots
    for i in range(1, 5):
        # Control medians
        plt.text(x[i - 1] - width/2, 
                 control_median[f'box_{i}_solve_time'] - 15,  # Position just below the dot
                 f'{control_median[f"box_{i}_solve_time"]:.1f}', 
                 ha='center', va='top', fontsize=10, color='darkblue')
        
        # HFS medians
        plt.text(x[i - 1] + width/2, 
                 hfs_median[f'box_{i}_solve_time'] - 15,  # Position just below the dot
                 f'{hfs_median[f"box_{i}_solve_time"]:.1f}', 
                 ha='center', va='top', fontsize=10, color='darkred')
    
    # Customize the plot
    plt.xlabel('Box Number', fontsize=14)
    plt.ylabel('Solve Time (ms)', fontsize=14)
    plt.title('Box Solve Times: Averages and Medians (HFS vs Control)', fontsize=16)
    plt.xticks(x, [f'Box {i}' for i in range(1, 5)], fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels for averages
    def add_value_labels(values, offset, color):
        for i, v in enumerate(values):
            plt.text(i + offset, v + 10, f'{v:.1f}', ha='center', va='bottom', fontsize=10, color=color)
    
    add_value_labels([control_avg[f'box_{i}_solve_time'] for i in range(1, 5)], -width/2, 'blue')
    add_value_labels([hfs_avg[f'box_{i}_solve_time'] for i in range(1, 5)], width/2, 'red')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


    
def plot_solve_times_comparison_all_dots(results):
    # Get the average and median solve times
    hfs_avg = results['hfs_avg_solve_times']
    control_avg = results['control_avg_solve_times']
    hfs_median = results['hfs_median_solve_times']
    control_median = results['control_median_solve_times']
    
    # Get standard deviations
    hfs_std = results['hfs_trials'][[f'box_{i}_solve_time' for i in range(1, 5)]].std()
    control_std = results['control_trials'][[f'box_{i}_solve_time' for i in range(1, 5)]].std()
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Set the positions for the bars
    x = np.arange(4)  # 4 boxes
    width = 0.35  # width of the bars
    
    # Create bars for averages with error bars
    plt.bar(x - width/2, 
            [control_avg[f'box_{i}_solve_time'] for i in range(1, 5)],
            width, 
            label='Control Avg',
            color='royalblue',
            yerr=[control_std[f'box_{i}_solve_time'] for i in range(1, 5)],
            capsize=5)
    
    plt.bar(x + width/2, 
            [hfs_avg[f'box_{i}_solve_time'] for i in range(1, 5)],
            width, 
            label='HFS Avg',
            color='lightcoral',
            yerr=[hfs_std[f'box_{i}_solve_time'] for i in range(1, 5)],
            capsize=5)
    
    # Add scatter points for individual data
    for i in range(1, 5):
        # Scatter Control solve times
        plt.scatter(
            np.full(len(results['control_trials']), x[i - 1] - width/2),
            results['control_trials'][f'box_{i}_solve_time'],
            color='blue', alpha=0.6, label='Control Times' if i == 1 else "", s=30
        )
        # Scatter HFS solve times
        plt.scatter(
            np.full(len(results['hfs_trials']), x[i - 1] + width/2),
            results['hfs_trials'][f'box_{i}_solve_time'],
            color='red', alpha=0.6, label='HFS Times' if i == 1 else "", s=30
        )
    
    # Customize the plot
    plt.xlabel('Box Number', fontsize=14)
    plt.ylabel('Solve Time (ms)', fontsize=14)
    plt.title('Box Solve Times: Averages and Individual Data Points (HFS vs Control)', fontsize=16)
    plt.xticks(x, [f'Box {i}' for i in range(1, 5)], fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels for averages
    def add_value_labels(values, offset, color):
        for i, v in enumerate(values):
            plt.text(i + offset, v + 10, f'{v:.1f}', ha='center', va='bottom', fontsize=10, color=color)
    
    add_value_labels([control_avg[f'box_{i}_solve_time'] for i in range(1, 5)], -4*width/5 , 'blue')
    add_value_labels([hfs_avg[f'box_{i}_solve_time'] for i in range(1, 5)], 4*width/5, 'red')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
def main():
    print("Analyzing data...")
    try:
        results = process_monkey_trials(PATH_TO_DATA)
        plot_solve_times_comparison(results)
        # plot_solve_times_comparison_all_dots(results)
    except KeyError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
