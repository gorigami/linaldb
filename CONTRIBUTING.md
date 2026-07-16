# Contributing to LINAL

Thank you for your interest in contributing to LINAL! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Project Structure](#project-structure)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

---

## Getting Started

### Prerequisites

- Rust 1.70+ (check with `rustc --version`)
- Cargo (comes with Rust)
- Git

### Setting Up the Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:

   ```bash
   git clone https://github.com/yourusername/linal.git
   cd linal
   ```

3. **Add upstream remote**:

   ```bash
   git remote add upstream https://github.com/gorigami/linal.git
   ```

4. **Build the project**:

   ```bash
   cargo build
   ```

5. **Run tests**:

   ```bash
   cargo test
   ```

---

## Development Workflow

### 1. Create a Branch

Create a feature branch from `main`:

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Changes

- Write clean, readable code
- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git commit -m "Add feature: description of what you did"
```

Commit message format:

- Use imperative mood ("Add feature" not "Added feature")
- First line should be < 50 characters
- Add detailed description if needed (separated by blank line)

Example:

```text
Add vector similarity search optimization

- Implement approximate nearest neighbor search
- Add index-based filtering for large datasets
- Improve query performance by 10x for similarity queries
```

### 5. Keep Your Branch Updated

Regularly sync with upstream:

```bash
git fetch upstream
git rebase upstream/main
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Coding Standards

### Rust Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting:

  ```bash
  cargo fmt
  ```

- Use `clippy` for linting:

  ```bash
  cargo clippy -- -D warnings
  ```

### Code Organization

- **Core types** go in `src/core/`
- **Engine logic** goes in `src/engine/`
- **DSL parsing** goes in `src/dsl/`
- **Query planning** goes in `src/query/`
- **Server code** goes in `src/server/`

### Error Handling

- Use `Result<T, E>` for fallible operations
- Provide clear error messages
- Use `thiserror` for error types
- Propagate errors with `?` operator

### API Documentation

- Document public APIs with `///` comments
- Include examples in doc comments
- Update README/docs for user-facing changes

Example:

```rust
/// Computes the dot product of two vectors.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The dot product as a scalar value.
///
/// # Example
///
/// ```
/// use linal::ops::dot_1d;
/// let v1 = vec![1.0, 2.0, 3.0];
/// let v2 = vec![4.0, 5.0, 6.0];
/// let result = dot_1d(&v1, &v2);
/// assert_eq!(result, 32.0);
/// ```
pub fn dot_1d(a: &[f32], b: &[f32]) -> f32 {
    // ...
}
```

---

## Testing

### Test Structure

- **Unit tests**: In same file as code (in `#[cfg(test)]` module)
- **Integration tests**: In `tests/` directory
- **Example tests**: In `examples/` directory (run with `cargo test --examples`)

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        // Arrange
        let input = ...;
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected);
    }
}
```

### Test Coverage

- Aim for high test coverage
- Test edge cases and error conditions
- Test both success and failure paths

### Running Tests

```bash
# All tests
cargo test

# Specific test file
cargo test --test test_file_name

# With output
cargo test -- --nocapture

# Integration tests only
cargo test --test '*'
```

---

## Documentation

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Note any limitations or caveats

### User Documentation

- Update `README.md` for user-facing features
- Add examples to `examples/` directory
- Update `docs/DSL_REFERENCE.md` for DSL changes
- Update `docs/ARCHITECTURE.md` for architectural changes

### Documentation Format

- Use Markdown for documentation files
- Include code examples with syntax highlighting
- Keep documentation up-to-date with code changes

---

## Submitting Changes

### Pull Request Process

1. **Create a Pull Request** on GitHub.
2. **Automated CI**: GitHub Actions will automatically run `cargo test`, `fmt`, and `clippy`. Your PR **must pass** these checks to be eligible for merging.
3. **Fill out the PR template**:
   - Description of changes
   - Related issues (if any)
   - Testing performed
   - Breaking changes (if any)

4. **Ensure CI passes**:
   - All tests pass
   - Code is formatted (`cargo fmt`)
   - No clippy warnings (`cargo clippy`)

5. **Respond to feedback**:
   - Address review comments
   - Make requested changes
   - Keep discussion constructive

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CI Status Checks passing (Automated)
- [ ] At least 1 Approval from maintainers
- [ ] No clippy warnings
- [ ] Code is formatted
- [ ] Commit messages are clear

---

## Project Structure

```text
linal/
├── src/
│   ├── core/          # Core data structures
│   ├── engine/        # Execution engine
│   ├── dsl/           # DSL parsing and execution
│   ├── query/         # Query planning and optimization
│   ├── server/        # HTTP server
│   └── utils/         # Utility functions
├── tests/             # Integration tests
├── examples/          # Example scripts
├── docs/              # Documentation
│   ├── ARCHITECTURE.md
│   ├── DSL_REFERENCE.md
│   └── archive/        # Retired planning/roadmap docs (gitignored — local only, won't appear after a fresh clone)
├── Cargo.toml
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Areas for Contribution

### High Priority

- Performance optimizations
- Additional aggregation functions
- More index types
- Query optimization improvements

### Medium Priority

- Additional DSL commands
- More tensor operations
- Better error messages
- Documentation improvements

### Low Priority

- Code cleanup and refactoring
- Test coverage improvements
- Example scripts
- Documentation examples

---

## Getting Help

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` directory for detailed information

---

## License

By contributing to LINALDB, you agree that your contributions will be licensed under the **LinalDB Community License v1.0**.

---

Thank you for contributing to LINALDB! 🚀
