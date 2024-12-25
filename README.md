# Agent Framework

This repository provides a flexible system for building and orchestrating **agents** and **workflows**. It offers two modes:

- **Client Mode**: Where tasks call out to a remote API client (e.g., your `client.py` functions).  
- **Local Mode**: Where tasks run directly in the local environment, utilizing `run_agents(...)` and local logic.

It also supports loading **YAML or JSON** workflows to define multi-step tasks.

---

## Table of Contents

1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Concepts](#concepts)  
   - [Agents](#agents)  
   - [Tasks](#tasks)  
   - [Client Mode vs Local Mode](#client-mode-vs-local-mode)  
   - [Workflows (YAML/JSON)](#workflows-yamljson)  
4. [Usage](#usage)  
   - [Creating Agents](#creating-agents)  
   - [Building Tasks](#building-tasks)  
   - [Running a Local Workflow](#running-a-local-workflow)  
   - [Running a Remote Workflow (Client Mode)](#running-a-remote-workflow-client-mode)  
   - [Uploading YAML/JSON Workflows](#uploading-yamljson-workflows)  
5. [Examples](#examples)  
   - [Simple Summarize Task](#simple-summarize-task)  
   - [Chainable Workflows](#chainable-workflows)  
   - [Custom Workflow](#custom-workflow)  
   - [Loading From a YAML File](#loading-from-a-yaml-file)  
6. [API Endpoints](#api-endpoints)  
7. [License](#license)

---

## Overview

The framework has distilled Agents into 3 distinct pieces:
- **Agents**
- **Tasks**
- **Workflows**

The **Agent** can be configured with:

- **Model Provider** (e.g., OpenAI, Llama, etc.)  
- **Tools** (e.g., specialized functions)

Users can define tasks (like `council`, `sentiment`, `translate_text`, etc.) in a **local** or **client** mode. They can also upload workflows (in YAML or JSON) to orchestrate multiple steps in sequence.

---

## Installation

1. **Clone the Repo**:

    ```bash
    git clone https://github.com/webcoderz/agents-framework.git
    cd agents-framework
    ```

2. **Install Dependencies**:

    ```bash
    uv pip install -r requirements.txt
    ```

3. **Set Environment Variables**:
    - `OPENAI_API_KEY` for the default OpenAI-based `ChatOpenAI`.
    - `LOGGING_LEVEL` (optional) to configure logging verbosity: `DEBUG`, `INFO`, etc.

---

## Concepts

### Agents

- They can have a custom model provider (e.g., `ChatOpenAI`, a Llama-based model, etc.).
- Agents can have tools attached, which are specialized functions accessible during execution.

### Tasks

- A **task** is a single step in a workflow, e.g., `council`, `schedule_reminder`, `sentiment`, `translate_text`, etc.
- Tasks are managed by the `Tasks` class in `tasks.py`.
- Tasks can be chained for multi-step logic (e.g., `tasks(text="...").council().sentiment().run_tasks()`).

### Client Mode vs Local Mode

- **Local Mode**: The system calls `run_agents(...)` directly in your local environment.  
- **Client Mode**: The system calls out to remote endpoints in a separate API.
  - In `client_mode=True`, each task (e.g. `sentiment`) triggers a client function (`sentiment_analysis(...)`) instead of local logic.

This allows you to **switch** between running tasks locally or delegating them to a server.

### Workflows (YAML/JSON)

- You can define multi-step workflows in YAML or JSON.
- The endpoint `/upload-workflow` accepts a file (via multipart form data).
  - First tries parsing **JSON**.
  - If that fails, it tries **YAML**.
- The file is validated against a `WorkflowDefinition` Pydantic model.
- Each step has a `type` (e.g., `"sentiment"`, `"custom"`) and optional parameters (like `agents`, `target_language`, etc.).

---

## Usage

### Creating Agents

```python
from framework.src.agents import Agent

my_agent = Agent(
    name="MyAgent",
    instructions="You are a helpful agent.",
    model_provider="default"   # or use a callable for custom model
)
```

### Building Tasks

In Python code, you can create tasks by instantiating the Tasks class and chaining methods:


```python
from framework.src.tasks import Tasks

tasks = Tasks(text="This is the text to analyze", client_mode=False)
(
  tasks
    .sentiment(agents=[my_agent])
    .council()   # a second step
)

results = tasks.run_tasks()
print(results)
```
Because client_mode=False, everything runs locally.

### Running a Local Workflow

```python
tasks = Tasks(text="Breaking news: local sports team wins!", client_mode=False)
tasks.summarize_text(max_words=50).run_tasks()
```

### Running a Remote Workflow (Client Mode)

```python
tasks = Tasks(text="Breaking news: local sports team wins!", client_mode=True)
tasks.summarize_text(max_words=50).run_tasks()
```
Now, summarize_text calls the client function (e.g., summarize_task(...)) instead of local logic.

### Uploading YAML/JSON Workflows
	1.	Create a YAML or JSON file specifying tasks:

```yaml
name: "My YAML Workflow"
text: "Large text to analyze"
workflow:
  - type: "sentiment"
  - type: "summarize_text"
    max_words: 20
  - type: "moderation"
    threshold: 0.7
  - type: "custom"
    name: "special-step"
    objective: "Analyze the text"
    instructions: "Use advanced analysis"
    context:
      extra_info: "some metadata"
```

	2.	Upload via the /upload-workflow endpoint (multipart file upload).
The server reads it as JSON or YAML and runs the tasks sequentially in local mode.

## Examples

### Simple Summarize Task

```python
tasks = Tasks("Breaking news: new Python release!", client_mode=False)
tasks.summarize_text(max_words=30).run_tasks()
```

Returns a summarized result.

### Chainable Workflows

```python
tasks = Tasks("Tech giant acquires startup for $2B", client_mode=False)
(tasks
   .council()
   .translate_text(target_language="es")
   .sentiment()
)
results = tasks.run_tasks()
```

	1.	Council step,
	2.	Translate to Spanish,
	3.	Sentiment analysis.

### Custom Workflow
```python
tasks = Tasks("Analyze this special text", client_mode=False)
tasks.custom(
    name="my-unique-step",
    objective="Perform advanced analysis",
    instructions="Focus on entity extraction and sentiment",
    agents=[my_agent],
    **{"extra_context": "some_val"}
)
results = tasks.run_tasks()
```

A "custom" task can reference a custom function in the CUSTOM_WORKFLOW_REGISTRY or fall back to a default behavior.

### Loading From a YAML File

```bash
curl -X POST "http://<your server>/upload-workflow" \
     -F "yaml_file=@path/to/workflow.yaml"
```

## API Endpoints

Here are some of the key endpoints if you integrate via REST:  
   - POST /council: Runs a council vote with ScheduleRequest.task.  
   - POST /reasoning: Runs a reasoning step with TextRequest. 
   - POST /summarize: Summarizes text in TextRequest.
   - POST /sentiment: Performs sentiment analysis on TextRequest.
   - POST /extract-entities: Extracts categorized entities.
   - POST /translate: Translates text.
   - POST /classify: Classifies text.
   - POST /moderation: Moderation checks with a threshold.
   - POST /custom-workflow: Runs a single “custom” step from CustomWorkflowRequest.
   - POST /upload-workflow: Accepts JSON or YAML for multi-step workflows.
