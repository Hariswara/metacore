FROM golang:1.22-alpine AS build
WORKDIR /src
COPY . .
RUN go build -o /out/gating ./cmd/gating

FROM alpine:3.20
COPY --from=build /out/gating /usr/local/bin/gating
ENTRYPOINT ["gating"]
