#!/usr/bin/env python3
"""
Wrapper script to run validation and capture output.
"""

import subprocess
import sys
import os

def run_validation():
    """Run the validation script and capture output."""
    script_path = os.path.join("papaya_models", "validate_framework.py")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        print("=== VALIDATION OUTPUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== ERRORS ===")
            print(result.stderr)
        
        print(f"=== EXIT CODE: {result.returncode} ===")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running validation: {e}")
        return False

if __name__ == "__main__":
    success = run_validation()
    print(f"\nValidation {'PASSED' if success else 'FAILED'}")
