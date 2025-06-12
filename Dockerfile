FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app/

# Expose port 80 untuk Claw Cloud
EXPOSE 80

# Jalankan langsung dengan python (bukan flask run)
CMD ["python", "app.py"]
