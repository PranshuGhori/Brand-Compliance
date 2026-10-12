import logging

from backend.src.config import Settings

logger = logging.getLogger("frameguard.telemetry")


def setup_telemetry(settings: Settings) -> bool:
    """Send traces and logs to Azure Monitor when a connection string is configured."""
    if not settings.applicationinsights_connection_string:
        logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING not set; telemetry disabled")
        return False
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(
        connection_string=settings.applicationinsights_connection_string,
        logger_name="frameguard",
    )
    return True
