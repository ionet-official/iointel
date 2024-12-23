from typing import List

def schedule_reminder(self, delay: int = 0, agents: List[str]=None):
    #WIP
    self.tasks.append({
        "type": "schedule_reminder",
        "command": self.text,
        "delay": delay,
        "agents": agents
    })
    return self

def council(self):
    self.tasks.append({
        "type": "council",
        "task": self.text,
    })
    return self

def solve_with_reasoning(self, agents: List[str]=None):
    self.tasks.append({
        "type": "solve_with_reasoning",
        "goal": self.text,
        "agents": agents
    })
    return self

def summarize_text(self, max_words: int = 100, agents: List[str]=None):
    self.tasks.append({
        "type": "summarize_text",
        "text": self.text,
        "max_words": max_words,
        "agents": agents
    })
    return self

def sentiment(self, agents: List[str]=None):
    self.tasks.append({
        "type": "sentiment",
        "text": self.text,
        "agents": agents
    })
    return self

def extract_categorized_entities(self, agents: List[str]=None):
    self.tasks.append({
        "type": "extract_categorized_entities",
        "text": self.text,
        "agents": agents
    })
    return self

def translate_text(self, target_language: str, agents: List[str]=None):
    self.tasks.append({
        "type": "translate_text",
        "text": self.text,
        "target_language": target_language,
        "agents": agents
    })
    return self

def classify(self, classify_by: list, agents: List[str]=None):
    self.tasks.append({
        "type": "classify",
        "classify_by": classify_by,
        "to_be_classified": self.text,
        "agents": agents
    })
    return self

def moderation(self, threshold: float, agents: List[str]=None):
    self.tasks.append({
        "type": "moderation",
        "text": self.text,
        "threshold": threshold,
        "agents": agents
    })
    return self

def custom(self, name: str, objective: str, agents: list = None, instructions: str = "", **kwargs):
    """
    Allows users to define a custom workflow (or step) that can be chained
    like the built-in tasks. 'name' can help identify the custom workflow
    in run_tasks().
    
    :param name: Unique identifier for this custom workflow step.
    :param objective: The main objective or prompt for run_agents.
    :param agents: List of agents used (if None, a default can be used).
    :param instructions: Additional instructions for run_agents.
    :param kwargs: Additional data needed for this custom workflow.
    """
    self.tasks.append({
        "type": "custom",
        "name": name,
        "objective": objective,
        "agents": agents or [],
        "instructions": instructions,
        
        "kwargs": kwargs,  # store other data
    })
    return self


# Dictionary mapping method names to functions
CHAINABLE_METHODS = {
    "schedule_reminder": schedule_reminder,
    "council": council,
    "solve_with_reasoning": solve_with_reasoning,
    "summarize_text": summarize_text,
    "sentiment": sentiment,
    "extract_categorized_entities": extract_categorized_entities,
    "translate_text": translate_text,
    "classify": classify,
    "moderation": moderation,
    "custom": custom
}