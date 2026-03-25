# Contributing to CV Group Project

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

- Search existing issues before creating a new one
- Use a clear, descriptive title
- Include steps to reproduce the issue
- Attach screenshots or logs if applicable

### Pull Requests

1. **Fork the repository** and create a feature branch from `main`
2. **Keep your changes focused** - one feature or fix per PR
3. **Follow the existing code style** and structure
4. **Add tests** for new functionality (80%+ coverage required)
5. **Test your changes** before submitting: `pytest --cov=src --cov-fail-under=80`
6. **Update documentation** if needed

### Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CV_group-project.git
cd CV_group-project

# Create a virtual environment (Python 3.10 or 3.11 recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Download MediaPipe hand tracking model
# Place hand_landmarker.task in project root

# Run tests
pytest --cov=src --cov-fail-under=80

# Run the application
python run.py
```

### Code Quality Standards

- Write clear, readable code with appropriate comments
- Keep functions small and focused
- Handle errors explicitly
- Maintain 80%+ test coverage
- Log important events for debugging

### Project Structure

```
CV_group-project/
|-- src/           # Main source code
|-- tests/         # Test files
|-- scripts/       # Utility scripts
|-- docs/          # Documentation
|-- run.py         # Application entry point
|-- requirements.txt
|-- pytest.ini     # Test configuration
```

### Testing Guidelines

- Write unit tests for new functions
- Write integration tests for module interactions
- Run full test suite before submitting PR
- Coverage must not drop below 80%

## Questions?

Feel free to open an issue for questions about the project or development process.
