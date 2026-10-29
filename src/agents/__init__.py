"""Academic Helpdesk agents package."""

def __getattr__(name: str):
    """Load the coordinator only when root_agent is requested."""
    if name == "root_agent":
        from .academic_agent import root_agent
        return root_agent

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
