import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_all_tensorboard_data(log_dir, output_csv_path, scalar_tag='val/loss'):
    all_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'tfevents' in f]
    
    all_files.sort()
    
    print(f"Find {len(all_files)} log files in {log_dir}.")
    
    all_scalar_data = []

    for event_file in all_files:
        print(f"Loading file: {os.path.basename(event_file)} ...")
        try:
            ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
            ea.Reload()

            if scalar_tag in ea.Tags()['scalars']:
                scalars = ea.Scalars(scalar_tag)
                # scalars (wall_time, step, value)
                all_scalar_data.extend(scalars)
            else:
                print(f"  -> Warning: Tag '{scalar_tag}' not found in this file, skipped.")
                
        except Exception as e:
            print(f"  -> Error: Failed to load file: {e}")

    if not all_scalar_data:
        print("Error: No scalar data found.")
        return

    df = pd.DataFrame(all_scalar_data)

    df = df.sort_values(by='step')

    # Skip duplicated steps
    # df = df.drop_duplicates(subset='step', keep='last')

    df[['step', 'value']].to_csv(output_csv_path, index=False)
    print(f"Success! Data merged and saved to: {output_csv_path} (total {len(df)} records)")

extract_all_tensorboard_data('./checkpoints_adaptive/logs', 'ours_data.csv', scalar_tag='validation/main_loss')