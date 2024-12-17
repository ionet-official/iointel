from typing import List

def chain_schedule_reminder(self, command: str, delay: int = 0, agents: List[str]=None):
    self.tasks.append({
        "type": "schedule_reminder",
        "command": command,
        "delay": delay,
        "agents": agents
    })
    return self

def chain_council(self, task: str):
    self.tasks.append({
        "type": "council",
        "task": task,
    })
    return self

def chain_solve_with_reasoning(self, goal: str, agents: List[str]=None):
    self.tasks.append({
        "type": "solve_with_reasoning",
        "goal": goal,
        "agents": agents
    })
    return self

def chain_summarize_text(self, text: str, max_words: int = 100, agents: List[str]=None):
    self.tasks.append({
        "type": "summarize_text",
        "text": text,
        "max_words": max_words,
        "agents": agents
    })
    return self

def chain_sentiment(self, text: str, agents: List[str]=None):
    self.tasks.append({
        "type": "sentiment",
        "text": text,
        "agents": agents
    })
    return self

def chain_extract_categorized_entities(self, text: str, agents: List[str]=None):
    self.tasks.append({
        "type": "extract_categorized_entities",
        "text": text,
        "agents": agents
    })
    return self

def chain_translate_text(self, text: str, target_language: str, agents: List[str]=None):
    self.tasks.append({
        "type": "translate_text",
        "text": text,
        "target_language": target_language,
        "agents": agents
    })
    return self

def chain_classify(self, classify_by: list, to_be_classified: str, agents: List[str]=None):
    self.tasks.append({
        "type": "classify",
        "classify_by": classify_by,
        "to_be_classified": to_be_classified,
        "agents": agents
    })
    return self

def chain_moderation(self, text: str, threshold: float, agents: List[str]=None):
    self.tasks.append({
        "type": "moderation",
        "text": text,
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