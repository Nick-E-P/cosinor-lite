FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --all-extras --dev

COPY src ./src
COPY app.py ./app.py
COPY run.sh ./run.sh
COPY tests ./tests

EXPOSE 7860
CMD ["uv", "run", "python", "app.py"]
