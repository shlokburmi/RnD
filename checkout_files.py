import subprocess
import os
import sys
import time

def run_command(command):
    try:
        # Use a timeout for the subprocess to avoid indefinite hanging
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1

def main():
    print("Fetching file list from local index (git ls-files)...")
    stdout, stderr, returncode = run_command("git ls-files")
    if returncode != 0:
        print(f"Error listing files: {stderr}")
        return

    files = [line.strip() for line in stdout.split('\n') if line.strip()]
    
    print(f"Found {len(files)} tracked files. Starting incremental checkout...")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for i, file_path in enumerate(files):
        if os.path.exists(file_path):
            # Optional: check if size is 0 (failed previous checkout)
            if os.path.getsize(file_path) > 0:
                skipped_count += 1
                continue
            
        print(f"[{i+1}/{len(files)}] Checking out: {file_path}...", end='', flush=True)
        
        # Force checkout to overwrite empty/corrupt files
        cmd = f'git checkout origin/main -- "{file_path}"'
        stdout, stderr, ret = run_command(cmd)
        
        if ret == 0:
            print(" DONE")
            success_count += 1
        else:
            print(f" FAILED")
            print(f"Error: {stderr}")
            fail_count += 1
            # Simple retry logic
            print("Retrying...", end='', flush=True)
            time.sleep(2)
            stdout, stderr, ret = run_command(cmd)
            if ret == 0:
                print(" DONE (Retry)")
                success_count += 1
                fail_count -= 1
            else:
                 print(" FAILED (Retry)")
        
        # Small delay to prevent flooding if network is sensitive
        time.sleep(0.2)
            
    print("-" * 30)
    print(f"Checkout complete.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {skipped_count}")

if __name__ == "__main__":
    main()
