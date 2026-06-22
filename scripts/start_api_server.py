import os

from asc.api.auth import _init_default_store
from asc.api.server import create_app
from asc.core.cluster_config import ClusterConfig

if __name__ == "__main__":
    os.environ["ASC_ADMIN_API_KEY"] = "asc-admin-2026"
    os.environ["ASC_API_KEY"] = "asc-user-2026"
    os.environ["ASC_ALLOW_NO_AUTH"] = "0"

    _init_default_store()

    import uvicorn

    config = ClusterConfig()
    app = create_app(cluster_config=config)
    uvicorn.run(app, host="0.0.0.0", port=8080)
