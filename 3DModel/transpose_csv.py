import pandas as pd
import sys

def transpose_csv(input_file, output_file):
    """
    Transpose a CSV file from row-based format (data, aoa, cl, cd, cm) 
    to column-based format (aoa, cl_ufov4, cd_ufov4, cm_ufov4, cl_sinc, cd_sinc, cm_sinc, etc.)
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        print(f"Input file columns: {list(df.columns)}")
        print(f"Input file shape: {df.shape}")
        print(f"First few rows:")
        print(df.head())
        
        # Check required columns
        required_cols = ['data', 'aoa', 'cl', 'cd', 'cm']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return
        
        # Extract object names from the data column (remove .obj extension)
        df['object'] = df['data'].str.replace('.obj', '')
        
        # Get unique objects and AOA values
        objects = sorted(df['object'].unique())
        aoa_values = sorted(df['aoa'].unique())
        
        print(f"Objects found: {objects}")
        print(f"AOA values: {len(aoa_values)} values from {min(aoa_values)} to {max(aoa_values)}")
        
        # Create the transposed structure
        result_data = []
        
        for aoa in aoa_values:
            row = {'aoa': aoa}
            
            # For each object, add its cl, cd, cm values for this AOA
            for obj in objects:
                # Find the row for this object and AOA
                obj_data = df[(df['object'] == obj) & (df['aoa'] == aoa)]
                
                if not obj_data.empty:
                    # Add columns for this object
                    row[f'cl_{obj}'] = obj_data.iloc[0]['cl']
                    row[f'cd_{obj}'] = obj_data.iloc[0]['cd']
                    row[f'cm_{obj}'] = obj_data.iloc[0]['cm']
                else:
                    # If no data for this combination, fill with NaN
                    row[f'cl_{obj}'] = None
                    row[f'cd_{obj}'] = None
                    row[f'cm_{obj}'] = None
            
            result_data.append(row)
        
        # Create DataFrame from result
        result_df = pd.DataFrame(result_data)
        
        # Create the desired column order: aoa, then for each object: cl, cd, cm
        column_order = ['aoa']
        for obj in objects:
            column_order.extend([f'cl_{obj}', f'cd_{obj}', f'cm_{obj}'])
        
        # Reorder columns
        result_df = result_df.reindex(columns=column_order)
        
        # Sort by AOA
        result_df = result_df.sort_values('aoa')
        
        # Write to output file
        result_df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully transposed {input_file} to {output_file}")
        print(f"Original shape: {df.shape}")
        print(f"Transposed shape: {result_df.shape}")
        print(f"\nOutput columns: {list(result_df.columns)}")
        print(f"First few rows of output:")
        print(result_df.head())
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to handle command line arguments or use default filenames.
    """
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = input_file.replace('.csv', '_transposed.csv')
    else:
        # Default filenames
        input_file = 'input.csv'
        output_file = 'output.csv'
        print(f"No arguments provided. Using default files: {input_file} -> {output_file}")
    
    transpose_csv(input_file, output_file)

if __name__ == "__main__":
    main()
