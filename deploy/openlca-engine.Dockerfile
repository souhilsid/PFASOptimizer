FROM ghcr.io/greendelta/gdt-server-app AS app
FROM ghcr.io/greendelta/gdt-server-lib AS lib
FROM ghcr.io/greendelta/gdt-server-native AS native

FROM eclipse-temurin:21-jre

ENV JAVA_MAX_RAM_PERCENTAGE=80

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=app /app /app
COPY --from=lib /app/lib /app/lib
COPY --from=native /app/native /app/native

RUN chmod +x /app/run.sh

# Mount or copy an openLCA workspace at /app/data:
#   /app/data/databases/Biochar
#   /app/data/libraries
#
# For Render, attach a persistent disk at /app/data and upload the database
# folder there, or build a private image that includes deploy/openlca-data.

EXPOSE 8080

ENTRYPOINT ["/app/run.sh"]
CMD ["-data", "/app/data", "-db", "Biochar", "--readonly", "-port", "8080"]
