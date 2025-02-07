import os


def _patch_openai_init():
    # lo and behold, here's some magic of Python monkeypatching
    # to silence a warning from controlflow about
    # "controlflow.llm.models - The default LLM model could not be created"
    from langchain_openai import ChatOpenAI

    orig_init = ChatOpenAI.__init__

    def patched_init(*args, **kw):
        try:
            return orig_init(*args, **kw)
        except Exception:
            return None

    ChatOpenAI.__init__ = patched_init
    # trigger the call to create default model
    import controlflow.llm.models  # noqa: F401

    ChatOpenAI.__init__ = orig_init


UNUSED = _patch_openai_init()

# turn off most prefect log messages, as they aren't useful
# to end user, but might hurt UX for inexperience ones
os.environ["PREFECT_LOGGING_LEVEL"] = os.environ.get(
    "PREFECT_LOGGING_LEVEL", "CRITICAL"
)
