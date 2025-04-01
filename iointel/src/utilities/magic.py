import os


# turn off most prefect log messages, as they aren't useful
# to end user, but might hurt UX for inexperienced ones
for env_name in ("PREFECT_LOGGING_SERVER_LEVEL", "PREFECT_LOGGING_LEVEL"):
    os.environ[env_name] = os.environ.get(env_name, "CRITICAL")

UNUSED = None
