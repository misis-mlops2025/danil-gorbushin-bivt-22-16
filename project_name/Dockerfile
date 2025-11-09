FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY project_name /app/project_name

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir .

CMD ["python", "-m", "project_name.modeling.train", "--help"]
