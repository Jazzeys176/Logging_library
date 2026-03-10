from .sdk import SDKTracer
from .transport import Telemetry

def init(backend_url=None, cosmos_conn=None, db_name=None, container_name=None):
    """Initializes and returns a tracer instance."""
    telemetry = Telemetry(
        backend_url=backend_url,
        cosmos_conn=cosmos_conn,
        db_name=db_name,
        container_name=container_name
    )
    return SDKTracer(telemetry)

__all__ = ["SDKTracer", "Telemetry", "init"]
