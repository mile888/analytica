## Quick Start

```bash
# 1. Clon a repository (e.g. via SSH)
git clone git@github.com:mile888/analytica.git
cd analytica

# 2. Enviroment setup
make setup

# 3. Dependency update
make update

# 4. Create .env file and add keys
cp .env_example .env

# 5. Run app
make run

# 6. To install a new dependency
uv add package-name
```

---
