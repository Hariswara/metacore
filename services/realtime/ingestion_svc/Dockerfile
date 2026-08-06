FROM golang:1.22-alpine AS build
WORKDIR /src
COPY . .
RUN go build -o /out/ingest ./cmd/ingest

FROM alpine:3.20
COPY --from=build /out/ingest /usr/local/bin/ingest
ENTRYPOINT ["ingest"]
