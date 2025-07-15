import os
import subprocess

# Define directories
airfoils_dir = 'C:\\Users\\alexa\\Downloads\\coord_seligFmt\\coord_seligFmt'
polars_dir = 'C:\\Users\\alexa\\Downloads\\polar'
os.makedirs(polars_dir, exist_ok=True)


re = 1e6 
mach = 1
aoa_start = 0
aoa_end = 15
aoa_step = 0.25

def run_xfoil(dat_path, polar_path):
    # XFOIL commands: Load airfoil, set operating conditions, compute polar sequence
    commands = [
        'NORM',
        f'LOAD {dat_path}',
        'PANE',
        'OPER',
        f'VISC {re}', 
        f'MACH {mach}',
        'PACC',
        polar_path,
        '', 
        f'SEQP',
        'ASEQ', 
        str(aoa_start),
        str(aoa_end),
        str(aoa_step),
        '',
        'PACC',
        'QUIT'
        ]

    try:
        process = subprocess.Popen(['xfoil'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for cmd in commands:
            process.stdin.write(cmd + '\n')
        process.stdin.close()
        
        output, error = process.communicate()
        

        if process.returncode != 0 or error:
            print(f"Error running XFOIL for {dat_path}:")
            print(error)
            print(output)
            return False
        else:
            print(f"Generated polar: {polar_path}")
            return True
    except FileNotFoundError:
        print("XFOIL not found. Make sure it's installed and in your PATH.")
        return False


for dat_filename in os.listdir(airfoils_dir):
    if not dat_filename.endswith('.dat'):
        continue
    
    airfoil_name = dat_filename[:-4]
    dat_path = os.path.join(airfoils_dir, dat_filename)
    polar_path = os.path.join(polars_dir, f"{airfoil_name}_polar.txt")
    
    run_xfoil(dat_path, polar_path)