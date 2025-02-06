# lo and behold, here's some magic of Python monkeypatching
# to silence a warning from controlflow about
# "controlflow.llm.models - The default LLM model could not be created"


def _patch_openai_init():
    from langchain_openai import ChatOpenAI

    orig_init = ChatOpenAI.__init__

    def patched_init(*args, **kw):
        try:
            return orig_init(*args, **kw)
        except Exception:
            return None

    ChatOpenAI.__init__ = patched_init
    ChatOpenAI.__init__ = orig_init


UNUSED = _patch_openai_init()
