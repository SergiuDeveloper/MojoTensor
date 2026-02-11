#!/bin/bash
set -e

any_failed=false
while IFS= read -r test_file; do
    if ! pixi run mojo -I $(pwd) "$test_file"; then
        any_failed=true
    fi
done < <(find "tst" -name "test_*.mojo" -type f | sort)

if [ "$any_failed" = true ]; then
    exit 1
fi
