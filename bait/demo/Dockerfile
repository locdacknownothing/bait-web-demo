FROM python:3.10-slim

WORKDIR /app

COPY . /app

# RUN apt-get update && apt-get install -y rsync
# RUN chmod +x /app/bash/*.sh
# RUN /app/bash/download_weights.sh

RUN pip install -r /app/requirements.txt

EXPOSE 1234

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "/app/app.py"]
