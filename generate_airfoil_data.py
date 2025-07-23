import os
import subprocess
import pandas as pd
from tqdm import tqdm
import multiprocessing  # For parallel processing

# Directories
airfoils_dir = 'C:\\Users\\alexa\\Downloads\\coord_seligFmt\\coord_seligFmt'
polars_dir = r'C:\Users\alexa\Downloads\polar'
logs_dir = r'C:\Users\alexa\Downloads\polar_logs'
os.makedirs(polars_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

re = 1e6 
mach = 0.1

# Checkpoint setup
checkpoint_file = 'airfoil_processing_log.csv'
if os.path.exists(checkpoint_file):
    df_checkpoint = pd.read_csv(checkpoint_file)
else:
    df_checkpoint = pd.DataFrame(columns=['airfoil', 'status', 'polar_path', 'log_path', 'timestamp'])

def run_xfoil_worker(args):
    dat_filename, airfoils_dir, polars_dir, logs_dir = args
    airfoil_name = dat_filename[:-4]
    dat_path = dat_filename  # Relative, since cwd=airfoils_dir
    polar_path = os.path.join(polars_dir, f"{airfoil_name}_polar.txt").replace('\\', '/')
    log_path = os.path.join(logs_dir, f"{airfoil_name}_log.txt").replace('\\', '/')
    
    print(f"Running XFOIL for {dat_filename} with LOAD {dat_path} and polar {polar_path}")
    
    aoa_start = 0.0
    aoa_end = 12.0  # Reduced to avoid stall hangs; adjust if needed
    stride = 0.25
    
    commands = [
        'PLOP',
        'G F',
        '',
        f'LOAD {dat_path}',
        'PANE',
        'INIT',  # Re-init for convergence
        'OPER',
        'VPAR',
        'N 9.0',  # Ncrit=9 for better transition at Re=1e6
        '',
        'ITER 2000',  # Higher for tough cases
        'VACC 0.00001',
        f'VISC {re}',
        f'MACH {mach}',
        'PACC',
        polar_path,
        '',
        f'ASEQ {aoa_start} {aoa_end} {stride}',
        '',
        'PACC',  # Off
        '', '', 'QUIT', ''
    ]
    
    try:
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(['xfoil'], stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.STDOUT, text=True, cwd=airfoils_dir)
            for cmd in commands:
                process.stdin.write(cmd + '\n')
            process.stdin.close()
            
            process.wait(timeout=600)
            
            if process.returncode != 0:
                return False
            
            if os.path.exists(polar_path) and os.path.getsize(polar_path) > 500:
                return True
            return False
    
    except subprocess.TimeoutExpired:
        process.kill()
        return False  # Partial may exist
    except Exception as e:
        print(f"Error for {dat_filename}: {e}")
        return False

if __name__ == '__main__':  # Windows multiprocessing guard—prevents recursive imports!
    processed_airfoils = set(df_checkpoint[df_checkpoint['status'] == 'success']['airfoil']) if 'df_checkpoint' in globals() else set()

    all_rows = []
    airfoil_list = [f for f in os.listdir(airfoils_dir) if f.endswith('.dat')]

    # Parallel processing
    with multiprocessing.Pool(4) as pool:  # 4 cores; adjust based on your CPU
        args_list = [(f, airfoils_dir, polars_dir, logs_dir) for f in airfoil_list if f not in processed_airfoils]
        successes = list(tqdm(pool.imap(run_xfoil_worker, args_list), total=len(args_list), desc="Processing Airfoils (Parallel)"))

    for i, dat_filename in enumerate([f for f in airfoil_list if f not in processed_airfoils]):
        success = successes[i]
        status = 'success' if success else 'failed'
        polar_path = os.path.join(polars_dir, f"{dat_filename[:-4]}_polar.txt")
        log_path = os.path.join(logs_dir, f"{dat_filename[:-4]}_log.txt")
        new_row = pd.DataFrame({
            'airfoil': [dat_filename],
            'status': [status],
            'polar_path': [polar_path],
            'log_path': [log_path],
            'timestamp': [pd.Timestamp.now()]
        })
        df_checkpoint = pd.concat([df_checkpoint, new_row], ignore_index=True)
        df_checkpoint.to_csv(checkpoint_file, index=False)
        
        if os.path.exists(polar_path) and os.path.getsize(polar_path) > 500:
            try:
                df_polar = pd.read_csv(polar_path, skiprows=12, sep=r'\s+', names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
                for _, row in df_polar.iterrows():
                    all_rows.append([dat_filename, row['alpha'], row['CL']])
            except Exception as e:
                print(f"Parse error for {polar_path}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows, columns=['file_path', 'aoa', 'cl'])
        df.to_csv('airfoil_data.csv', index=False)
        print("Saved full dataset to airfoil_data.csv")
    else:
        print("No data collected—check logs/checkpoint for fails.")
