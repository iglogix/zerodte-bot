#!/bin/bash
#
# Setup script for 0DTE Trading Bot git hooks
#
# This script configures git to use the custom hooks directory
# which includes the pre-commit hook that protects tests_locked/
#

echo "Setting up git hooks for 0DTE Trading Bot..."

# Configure git to use .githooks directory
git config core.hooksPath .githooks

echo "Git hooks configured successfully!"
echo ""
echo "The pre-commit hook will now prevent modifications to tests_locked/"
echo "To bypass (with approval): git commit --no-verify"
