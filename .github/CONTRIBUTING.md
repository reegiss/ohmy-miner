# Contributing to OhMy Miner

Thank you for considering contributing to OhMy Miner! This document outlines the process for contributing to the project.

## Development Process

1. Fork the repository
2. Create a new branch for your feature/bugfix
3. Make your changes
4. Write/update tests as needed
5. Update documentation if required
6. Submit a pull request

## Building from Source

### Prerequisites

- CMake 3.15+
- C++20 compatible compiler
- CUDA Toolkit
- Git

### Build Instructions

```bash
mkdir build && cd build
cmake ..
make
```

## Testing

Run the tests using:
```bash
cd build
ctest
```

## Code Style

- Use modern C++ features (C++20)
- Follow consistent indentation (4 spaces)
- Add comments for complex logic
- Write descriptive commit messages

## Submitting Changes

1. Create a branch with a descriptive name
2. Make focused commits with clear messages
3. Write or update tests for your changes
4. Update documentation as needed
5. Submit a pull request with a clear description of changes

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

Feel free to open an issue for any questions about contributing!