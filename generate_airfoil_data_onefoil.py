#!/usr/bin/env python3
"""
XFOIL Runner Script

This script runs XFOIL on an airfoil data file with predefined analysis parameters.
Usage: python xfoil_runner.py <airfoil_data_file.dat>
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path

def run_xfoil_analysis(dat_path, foilname):
    """
    Run XFOIL analysis on the specified airfoil data file.
    
    Args:
        dat_path (str): Path to the airfoil data file (.dat format)
    
    Returns:
        tuple: (success, polar_data, error_message)
    """
    
    # Analysis parameters
    re = 6e6
    mach = 0.262
    aoa_start = 0.0
    aoa_end = 12.0
    stride = 1
    
    # Create temporary file for polar data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pol', delete=True) as tmp_file:
        polar_path = tmp_file.name
    
    try:
        # Build XFOIL command sequence
        commands = [
            f'LOAD {dat_path}',
            f'{foilname}',
            'PANE',
            'INIT',
            'OPER',
            'VPAR',
            'N 9.0',
            '',
            'ITER 5000',
            'VACC 0.00001',
            f'VISC {re}',
            f'MACH {mach}',
            'PACC',
            polar_path,
            '',
            f'ASEQ {aoa_start} {aoa_end} {stride}',
            '',
            'PACC',
            '', '', 'QUIT', ''
        ]
        
        # Join commands with newlines
        input_commands = '\n'.join(commands)
        
        print(f"Running XFOIL analysis on: {dat_path}")
        print(f"Reynolds number: {re}")
        print(f"Mach number: {mach}")
        print(f"Angle of attack range: {aoa_start}° to {aoa_end}° (step: {stride}°)")
        print(f"Output polar file: {polar_path}")
        print("-" * 60)
        
        # Run XFOIL
        process = subprocess.run(
            ['Xfoil/bin/xfoil'],
            input=input_commands,
            text=True,
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
        
        # Check if polar file was created and has data
        if os.path.exists(polar_path) and os.path.getsize(polar_path) > 0:
            with open(polar_path, 'r') as f:
                polar_data = f.read()
            
            print("XFOIL analysis completed successfully!")
            print(f"Polar data saved to: {polar_path}")
            
            # Display first few lines of results
            lines = polar_data.strip().split('\n')
            print("\nFirst few lines of polar data:")
            for line in lines[:10]:
                print(line)
            
            if len(lines) > 10:
                print(f"... and {len(lines) - 10} more lines")
            
            return True, polar_data, None
            
        else:
            error_msg = "XFOIL failed to generate polar data"
            if process.stderr:
                error_msg += f"\nSTDERR: {process.stderr}"
            if process.stdout:
                error_msg += f"\nSTDOUT: {process.stdout}"
            
            return False, None, error_msg
            
    except subprocess.TimeoutExpired:
        return False, None, "XFOIL analysis timed out (>5 minutes)"
    
    except FileNotFoundError:
        return False, None, "XFOIL executable not found. Please ensure XFOIL is installed and in PATH."
    
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"
    
    finally:
        # Clean up temporary file if it exists
        try:
            if os.path.exists(polar_path):
                # Optionally keep the polar file - comment out next line to preserve
                # os.unlink(polar_path)
                pass
        except:
            pass

def main():
    """Main function to handle command line arguments and run analysis."""
    
    if len(sys.argv) != 3:
        print("Usage: python xfoil_runner.py <airfoil_data_file.dat> foilname")
        print("\nExample: python xfoil_runner.py naca0012.dat foilname")
        sys.exit(1)
    
    dat_file = sys.argv[1]
    foilname = sys.argv[2]
    
    # Check if input file exists
    if not os.path.isfile(dat_file):
        print(f"Error: Airfoil data file '{dat_file}' not found.")
        sys.exit(1)
    
    # Check file extension
    if not dat_file.lower().endswith('.dat'):
        print("Warning: Input file doesn't have .dat extension. Proceeding anyway...")
    
    # Run the analysis
    success, polar_data, error_msg = run_xfoil_analysis(dat_file, foilname)
    
    if success:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Optionally save polar data to a named file
        output_file = Path(dat_file).stem + "_polar.txt"
        try:
            with open(output_file, 'w') as f:
                f.write(polar_data)
            print(f"Polar data also saved to: {output_file}")
        except Exception as e:
            print(f"Could not save to {output_file}: {e}")
        
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("ANALYSIS FAILED")
        print("="*60)
        print(f"Error: {error_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
