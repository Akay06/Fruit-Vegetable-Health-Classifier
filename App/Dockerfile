FROM python:3.9

WORKDIR /app

COPY ./backend /backend

RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY ./frontend /frontend

COPY ./models /models

EXPOSE 9000

CMD ["python", "/backend/FruitVegetableClassification.py"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["fastapi", "run", "app/main.py", "--port", "80", "--proxy-headers"]