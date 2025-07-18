import os
import subprocess
import pandas as pd

# Define directories
airfoils_dir = 'C:\\Users\\alexa\\Downloads\\coord_seligFmt\\coord_seligFmt'
polars_dir = r'C:\Users\alexa\Downloads\polar'
os.makedirs(polars_dir, exist_ok=True)

re = 1e6 
mach = 0.1

def run_xfoil(dat_path, polar_path, aoa):
    commands = [
        f'LOAD {dat_path}',
        'PANE',
        'PLOP',  # Enter plotting options
        'G F',   # Set graphics enable to false (suppress plots)
        '',      # Blank line to exit PLOP menu
        'OPER',
        f'ITER 10000',  # Lowered to speed up failures
        f'VISC {re}', 
        f'MACH {mach}',
        'PACC',
        polar_path,
        '', 
        f'ALFA {aoa}',
        '',
        '',
        'QUIT'
        ]
    # Get old file size before running
    old_size = os.path.getsize(polar_path) if os.path.exists(polar_path) else 0

    try:
        process = subprocess.Popen(['xfoil'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for cmd in commands:
            process.stdin.write(cmd + '\n')
        process.stdin.close()
        
        # Add timeout to prevent hangs
        try:
            output, error = process.communicate(timeout=30)  # 30-second timeout
        except subprocess.TimeoutExpired:
            process.kill()
            output, error = process.communicate()  # Clean up
            print(f"Timeout occurred for AoA {aoa} - assuming convergence failed {dat_path}")
            return False
        
        #print(output)
        
        if process.returncode != 0 or error:
            print(f"Error running XFOIL for {dat_path}:")
            print(error)
            print(output)
            return False
        
        # Check if new data was added (file size increased)
        new_size = os.path.getsize(polar_path)
        if new_size > old_size:
            print(f"Generated polar point for AoA {aoa}")
            return True
        else:
            print(f"No new data added for AoA {aoa}, assuming convergence failed for {dat_path}")
            return False
    
    except FileNotFoundError:
        print("XFOIL not found. Make sure it's installed and in your PATH.")
        return False

all_rows = []

for dat_filename in os.listdir(airfoils_dir):
    if not dat_filename.endswith('.dat'):
        continue
    airfoil_name = dat_filename[:-4]
    dat_path = os.path.join(airfoils_dir, dat_filename)
    polar_path = os.path.join(polars_dir, f"{airfoil_name}_polar.txt")
    if os.path.exists(polar_path):
        os.remove(polar_path)
    aoa = 0.0
    stride = 0.25
    max_aoa = 30.0  # Safety cap
    conv = True
    while aoa <= max_aoa and conv:
        conv = run_xfoil(dat_path, polar_path, aoa)
        if conv:
            aoa += stride
    # Parse polar after loop
    if os.path.exists(polar_path):
        try:
            df_polar = pd.read_csv(polar_path, skiprows=12, sep=r'\s+', names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
            for _, row in df_polar.iterrows():
                all_rows.append([dat_filename, row['alpha'], row['CL']])
        except Exception as e:
            print(f"Error parsing {polar_path}: {e}")

if all_rows:
    df = pd.DataFrame(all_rows, columns=['file_path', 'aoa', 'cl'])
    df.to_csv('airfoil_data.csv', index=False)
    print("Saved data to airfoil_data.csv")
else:
    print("No data collected.")