.PHONY: help fmt check test clippy build run clean doc all

# Default target
help:
	@echo "Available targets:"
	@echo "  make fmt      - Format code with rustfmt"
	@echo "  make check    - Check if code compiles"
	@echo "  make test     - Run all tests"
	@echo "  make clippy   - Run clippy lints"
	@echo "  make build    - Build the project"
	@echo "  make run      - Run the project"
	@echo "  make doc      - Generate documentation"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make all      - Format, check, test, clippy, and build"

# Format code
fmt:
	@echo "Formatting code..."
	cargo fmt

# Check formatting
fmt-check:
	@echo "Checking code formatting..."
	cargo fmt -- --check

# Quick compile check
check:
	@echo "Checking if code compiles..."
	cargo check --all-targets

# Run tests
test:
	@echo "Running tests..."
	cargo test

# Run clippy lints
clippy:
	@echo "Running clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

# Build release
build:
	@echo "Building project..."
	cargo build --release

# Build debug
build-dev:
	@echo "Building project (debug)..."
	cargo build

# Run the project
run:
	@echo "Running project..."
	cargo run --release

# Run in development mode
run-dev:
	@echo "Running project (debug)..."
	cargo run

# Generate documentation
doc:
	@echo "Generating documentation..."
	cargo doc --no-deps --open

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean

# Run all quality checks
all: fmt check test clippy build
	@echo "✅ All checks passed!"

# Pre-commit checks (same as CI)
ci: fmt-check clippy test
	@echo "✅ CI checks passed!"
