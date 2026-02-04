# Contributing to Patient Treatment Outcome Prediction

Thank you for your interest in contributing! 🎉

## How to Contribute

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/Patient-Treatment-Outcome-Prediction-MLOps-Project.git
cd Patient-Treatment-Outcome-Prediction-MLOps-Project
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation if needed

### 5. Test Your Changes

```bash
# Reproduce the pipeline
dvc repro

# Run the API locally
export API_KEY="test-token"
uvicorn src.api.main:app --reload
```

### 6. Submit a Pull Request

Push your branch and open a PR on GitHub.

## Code Style

- **Python**: Follow PEP 8 guidelines
- **Commits**: Use descriptive commit messages
- **Documentation**: Update README for user-facing changes

## Reporting Issues

Open an issue on GitHub with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior

## Questions?

Feel free to open a discussion on GitHub!
