FROM python:3.13-slim

WORKDIR /app

COPY . .

ARG PIP_INSECURE_TEST=true
RUN if [ "$PIP_INSECURE_TEST" = "true" ]; then \
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt; \
    else \
    pip install -r requirements.txt; \
    fi

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]