#!/bin/bash

# Check system tools
echo "Checking dependencies:"
echo -n "Homebrew: " && which brew >/dev/null && echo "OK" || echo "NOT FOUND"
echo -n "LLVM: " && brew list llvm &>/dev/null && echo "OK" || echo "NOT FOUND"

# Check Python environment
echo -n "Python 3.9: " && python3.9 --version | grep "3.9" && echo "OK" || echo "WRONG VERSION"
echo -n "Virtual Env: " && [[ $VIRTUAL_ENV == *"ddsp-env"* ]] && echo "ACTIVE ($VIRTUAL_ENV)" || echo "INACTIVE"

# Check packages using environment's Python
echo -e "\nChecking Python packages:"
packages=("tensorflow" "ddsp" "librosa" "numba" "llvmlite")
for pkg in "${packages[@]}"; do
    echo -n "$pkg: " 
    python -c "import $pkg; print('OK')" 2>/dev/null || echo "NOT FOUND"
done