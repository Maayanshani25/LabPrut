import pandas as pd
import numpy as np
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
            
            # Calculate solve time if all relevant columns are valid
            if (number_box_col in row.index and go_col in row.index and stop_col in row.index):
                if pd.notna(row[number_box_col]) and pd.notna(row[go_col]) and pd.notna(row[stop_col]):
                    box_num = int(row[number_box_col])
                    solve_times[f'box_{box_num}_solve_time'] = row[stop_col] - row[go_col]
        
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
    
    # Calculate average solve times
    hfs_avg_solve_times = hfs_trials[box_solve_cols].mean()
    control_avg_solve_times = control_trials[box_solve_cols].mean()
    
    # Statistical summaries
    hfs_solve_times_summary = hfs_trials[box_solve_cols].describe()
    control_solve_times_summary = control_trials[box_solve_cols].describe()
    
    return {
        'full_dataframe': df_processed,
        'hfs_trials': hfs_trials,
        'control_trials': control_trials,
        'hfs_avg_solve_times': hfs_avg_solve_times,
        'control_avg_solve_times': control_avg_solve_times,
        'hfs_solve_times_summary': hfs_solve_times_summary,
        'control_solve_times_summary': control_solve_times_summary
    }

# Example of reading data
def readData(filePath: str):
    df = pd.read_csv(filePath, sep=None, engine='python')
    print("Column names:", df.columns)
    print(df)

# Usage example
try:
    results = process_monkey_trials(PATH_TO_DATA)
    print("\nControl Average Solve Times:")
    print(results['control_avg_solve_times'])
    print()
    print("HFS Average Solve Times:")
    print(results['hfs_avg_solve_times'])
except KeyError as e:
    print(f"Error: {e}")
