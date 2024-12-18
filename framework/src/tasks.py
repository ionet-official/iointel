from typing import List

def chain_schedule_reminder(self, delay: int = 0, agents: List[str]=None):
    self.tasks.append({
        "type": "schedule_reminder",
        "command": self.text,
        "delay": delay,
        "agents": agents
    })
    return self

def chain_council(self):
    self.tasks.append({
        "type": "council",
        "task": self.text,
    })
    return self

def chain_solve_with_reasoning(self, agents: List[str]=None):
    self.tasks.append({
        "type": "solve_with_reasoning",
        "goal": self.text,
        "agents": agents
    })
    return self

def chain_summarize_text(self, max_words: int = 100, agents: List[str]=None):
    self.tasks.append({
        "type": "summarize_text",
        "text": self.text,
        "max_words": max_words,
        "agents": agents
    })
    return self

def chain_sentiment(self, agents: List[str]=None):
    self.tasks.append({
        "type": "sentiment",
        "text": self.text,
        "agents": agents
    })
    return self

def chain_extract_categorized_entities(self, agents: List[str]=None):
    self.tasks.append({
        "type": "extract_categorized_entities",
        "text": self.text,
        "agents": agents
    })
    return self

def chain_translate_text(self, target_language: str, agents: List[str]=None):
    self.tasks.append({
        "type": "translate_text",
        "text": self.text,
        "target_language": target_language,
        "agents": agents
    })
    return self

def chain_classify(self, classify_by: list, agents: List[str]=None):
    self.tasks.append({
        "type": "classify",
        "classify_by": classify_by,
        "to_be_classified": self.text,
        "agents": agents
    })
    return self

def chain_moderation(self, threshold: float, agents: List[str]=None):
    self.tasks.append({
        "type": "moderation",
        "text": self.text,
        "threshold": threshold,
        "agents": agents
    })
    return self


# Dictionary mapping method names to functions
CHAINABLE_METHODS = {
    "schedule_reminder": chain_schedule_reminder,
    "council": chain_council,
    "solve_with_reasoning": chain_solve_with_reasoning,
    "summarize_text": chain_summarize_text,
    "sentiment": chain_sentiment,
    "extract_categorized_entities": chain_extract_categorized_entities,
    "translate_text": chain_translate_text,
    "classify": chain_classify,
    "moderation": chain_moderation,
}