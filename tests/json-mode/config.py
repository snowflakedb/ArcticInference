"""Constants used for Corvo API calls, obtained from environment variable."""

import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "snowflake-arctic-embed-m-v1.5")
REQUEST_MAX_RETRIES = int(os.getenv("REQUEST_MAX_RETRIES", 10))
REQUEST_TIMEOUT_IN_SECONDS = int(os.getenv("REQUEST_TIMEOUT_IN_SECONDS", 30))

URL_CORVO_EMBED_XP = os.getenv(
    "URL_CORVO_EMBED_XP",
    "http://mixer.corvo.svc.cluster.local:8080/v1/embeddings_xp",
)
URL_CORVO_COMPLETE_XP = os.getenv(
    "URL_CORVO_COMPLETE_XP",
    "http://mixer.corvo.svc.cluster.local:8080/v1/textcompletion_xp",
)
