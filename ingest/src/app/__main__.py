from __future__ import annotations
import logging
from app.ingest_controller import ingest_many
from app.config import SETTINGS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    #TODO: for now just treat everything as standalone
    components = []
    #    {"repo": "frontend-app", "namespace": "frontend", "layer": "frontend", "collection": "product-alpha"},
    #    {"repo": "dmz-gateway", "namespace": "dmz", "layer": "middleware", "collection": "product-alpha"},
    #    {"repo": "service-a", "namespace": "service-a", "layer": "backend", "collection": "product-alpha"},
    #    # Standalone notebooks
    #    {"repo": "ml-notebooks", "namespace": "ml-notebooks", "layer": "standalone", "collection": "research", "component_kind": "standalone"},
    # ]

    print(ingest_many(components, branch=SETTINGS.default_branch, dev_force_standalone=SETTINGS.dev_force_standalone))
