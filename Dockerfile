FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements_dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "toxic_comment_classifier.train_model"]
