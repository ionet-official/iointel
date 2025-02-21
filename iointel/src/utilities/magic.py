import os


def _patch_openai_init():
    # lo and behold, here's some magic of Python monkeypatching
    # to silence a warning from controlflow about
    # "controlflow.llm.models - The default LLM model could not be created"
    from pydantic_ai.models.openai import OpenAIModel

    orig_init = OpenAIModel.__init__

    def patched_init(*args, **kw):
        try:
            return orig_init(*args, **kw)
        except Exception:
            return None

    OpenAIModel.__init__ = patched_init


    OpenAIModel.__init__ = orig_init


# turn off most prefect log messages, as they aren't useful
# to end user, but might hurt UX for inexperienced ones
for env_name in ("PREFECT_LOGGING_SERVER_LEVEL", "PREFECT_LOGGING_LEVEL"):
    os.environ[env_name] = os.environ.get(env_name, "CRITICAL")

UNUSED = _patch_openai_init()
