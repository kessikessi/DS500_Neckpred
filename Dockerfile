# Container image for the dashboard. Runs the same on a laptop and on bwCloud.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/health', timeout=3)"

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--access-logfile", "-", "app.main:server"]
