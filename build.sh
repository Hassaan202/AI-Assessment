#!/bin/bash
set -e

# Debug information
echo "Starting build script"
echo "Python version:"
python --version

# Ensure pip, setuptools, and wheel are available
echo "Installing base dependencies..."
pip install --upgrade pip setuptools wheel distutils-pytest

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

echo "Build completed successfully"