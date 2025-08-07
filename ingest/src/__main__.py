from __future__ import annotations
import logging
from ingest_controller import ingest_many
from .config import SETTINGS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    components = [
        {"repo": "frontend-app", "namespace": "frontend", "layer": "frontend", "collection": "product-alpha"},
        {"repo": "dmz-gateway", "namespace": "dmz", "layer": "middleware", "collection": "product-alpha"},
        {"repo": "service-a", "namespace": "service-a", "layer": "backend", "collection": "product-alpha"},
        # Standalone notebooks
        {"repo": "ml-notebooks", "namespace": "ml-notebooks", "layer": "standalone", "collection": "research", "component_kind": "standalone"},
    ]

    print(ingest_many(components, branch=SETTINGS.default_branch, dev_force_standalone=SETTINGS.dev_force_standalone))
