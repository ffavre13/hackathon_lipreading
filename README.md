# hackathon_lipreading

## Text inference
Infos are in : [src/vsr/README.md](src/vsr/README.md).

docker build -t lipreading .
docker run --env-file .env  -p 8080:8080 
docker compose up --build