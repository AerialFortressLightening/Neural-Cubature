import os
import subprocess
import sys
import time
import argparse
import re

def run_command(command, description):
    print(f"--- started {description} ---")
    print(f"Command: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8'
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc == 0:
            print(f"\n--- {description}succeed ---")
        else:
            print(f"\n--- {description}failed, code: {rc} ---")
        
        return rc
    except FileNotFoundError:
        print(f"Error: no file")
        return -1
    except Exception as e:
        print(f"Error while processing: {e}")
        return -1

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    parser = argparse.ArgumentParser(description="Greedy cubature for each displacement file")
    parser.add_argument("--pca_dim", type=int, default=227)
    parser.add_argument("--candidate_size", type=int, default=50)
    parser.add_argument("--save_intermediate", action="store_true")
    args = parser.parse_args()
    
    src_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(src_dir)
    
    data_dir = os.path.join(root_dir, "bunny_models/trajectory_data")
    pca_basis_dir = os.path.join(root_dir, "bunny_models/PCA_basis")
    initial_cubature_dir = os.path.join(root_dir, "bunny_models/initial_cubature")
    
    #os.makedirs(data_dir, exist_ok=True)
    #os.makedirs(pca_basis_dir, exist_ok=True)
    os.makedirs(initial_cubature_dir, exist_ok=True)
    
    python_executable = sys.executable
    
    start_time = time.time()
    
    dmat_files = [f for f in os.listdir(data_dir) if f.endswith('.dmat')]
    dmat_files.sort(key=natural_sort_key)
    if not dmat_files:
        print(f"None displacement file in {data_dir}")
        return

    for i, filename in enumerate(dmat_files):
        file_path = os.path.join(data_dir, filename)
        file_prefix = os.path.splitext(filename)[0] + '_'
        
        print(f"\n--- start processing {i+1}/{len(dmat_files)}: {filename} ---")
        greedy_params = {
            "system_name": "fem",
            "problem_name": "bunny",
            "data_file": file_path,
            "pca_basis": pca_basis_dir,
            "output_dir": initial_cubature_dir,
            "output_prefix": file_prefix,
            "max_n": 600,
            "tol_error": 0.00001,
            "n_candidate": args.candidate_size,
            "report_every": 50,
            "gradient_weighting": True,
            "save_intermediate": args.save_intermediate,
        }
        
        greedy_script_path = os.path.join(src_dir, "main_greedy_cubature.py")
        greedy_command = [python_executable, greedy_script_path]
        
        for key, value in greedy_params.items():
            if isinstance(value, bool) and value:
                greedy_command.append(f"--{key}")
            elif not isinstance(value, bool):
                greedy_command.append(f"--{key}")
                greedy_command.append(str(value))
        
        description = f"file {filename} in greedy cubature"
        greedy_result = run_command(greedy_command, description)
        
        if greedy_result != 0:
            print(f"{description} failed, terminate process.")
            return

    total_time = time.time() - start_time
    print(f"\n--- FINISH ---")
    print(f"TOTAL TIME: {total_time:.2f} sec")
    print(f"Files: {len(dmat_files)}")
    print(f"Candidate size: {args.candidate_size}")
    print(f"Saved path: {initial_cubature_dir}")

if __name__ == "__main__":
    main()
