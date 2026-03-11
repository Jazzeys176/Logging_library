from .sdk import SDKTracer
from .transport import Telemetry

def init(cosmos_conn=None, db_name=None, container_name=None, environment="prod", model="unknown", provider="unknown"):
    """Initializes and returns a tracer instance."""
    telemetry = Telemetry(
        cosmos_conn=cosmos_conn,
        db_name=db_name,
        container_name=container_name
    )
    return SDKTracer(telemetry, environment=environment, model=model, provider=provider)

__all__ = ["SDKTracer", "Telemetry", "init"]
