#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

pids=()
files=()

while IFS= read -r test_file; do
    pixi run mojo -I $(pwd) "$test_file" &
    pids+=($!)
    files+=("$test_file")
done < <(find "tst" -name "test_*.mojo" -type f | sort)

failed=()
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        failed+=("${files[$i]}")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for f in "${failed[@]}"; do
        echo -e "  ${RED}✗ $f${NC}"
    done
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
fi
