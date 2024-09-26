#!/bin/bash

# Define bold color codes
BOLD_YELLOW='\033[1;33m'
BOLD_RED='\033[1;31m'
BOLD_GREEN='\033[1;32m'
BOLD='\033[1m'
NC='\033[0m' # No Color (reset)

CODE_PATH=$1

echo -e "${BOLD}SUSPECT IDENTIFIED${NC}: $CODE_PATH"

# Run Pylint
echo -e "${BOLD_YELLOW}CHECKING SUSPECT WITH PYLINT${NC}"
pylint $CODE_PATH
if [ $? -ne 0 ]; then
    echo -e "${BOLD_RED}PYLINT CHECK FAILED!${NC}"
    exit 1
fi

# Run Flake8
echo -e "${BOLD_YELLOW}CHECKING SUSPECT WITH FLAKE8${NC}"
flake8 $CODE_PATH --append-config ~/.config/flake8/.flake8
if [ $? -ne 0 ]; then
    echo -e "${BOLD_RED}FLAKE8 CHECK FAILED!${NC}"
    exit 1
fi

# If all checks pass
echo -e "${BOLD_GREEN}ALL CHECKS PASSED!${NC}"
