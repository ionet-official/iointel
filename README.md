# IO Intelligence Agent Framework

> [!IMPORTANT]
> **Beta Notice:** This project is in rapid development and may not be stable for production use.

This repository provides a flexible system for building and orchestrating **agents** and **workflows**. It offers two modes:

- **Client Mode**: Where tasks call out to a remote API client (e.g., your `client.py` functions).  
- **Local Mode**: Where tasks run directly in the local environment, utilizing `run_agents(...)` and local logic.

It also supports loading **YAML or JSON** workflows to define multi-step tasks.

---

## Table of Contents<a id="table-of-contents"></a>

1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Concepts](#concepts)  
   - [Agents](#agents)  
   - [Tasks](#tasks)  
   - [Client Mode vs Local Mode](#client-mode-vs-local-mode)  
   - [Workflows (YAML/JSON)](#workflows-yamljson)  
4. [Usage](#usage)  
   - [Creating Agents](#creating-agents)  
   - [Creating an Agent with custom Persona](#creating-an-agent-with-a-persona)  
   - [Building a Workflow](#building-a-workflow)  
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

## Overview<a id="overview"></a>

The framework has distilled Agents into 3 distinct pieces:
- **Agents**
- **Tasks**
- **Workflows**

The **Agent** can be configured with:

- **Model Provider** (e.g., OpenAI, Llama, etc.)  
- **Tools** (e.g., specialized functions)

Users can define tasks (like `sentiment`, `translate_text`, etc.) in a **local** or **client** mode. They can also upload workflows (in YAML or JSON) to orchestrate multiple steps in sequence.

---

## Installation<a id="installation"></a>

1. **Install the latest release**:

  ```bash
  pip install --upgrade iointel
  ```

2. **For UI features (Gradio interface)**, install with UI dependencies:

  ```bash
  pip install --upgrade "iointel[ui]"
  ```

3. **Set Required Environment Variable**:
    - `OPENAI_API_KEY` or `IO_API_KEY` for the default OpenAI-based `ChatOpenAI`.

4. **Optional Environment Variables**:
    - `AGENT_LOGGING_LEVEL` (optional) to configure logging verbosity: `DEBUG`, `INFO`, etc.
    - `OPENAI_API_BASE_URL` or `IO_API_BASE_URL` to point to OpenAI-compatible API implementation, like `https://api.intelligence.io.solutions/api/v1`
    - `OPENAI_API_MODEL` or `IO_API_MODEL` to pick specific LLM model as "agent brain", like `meta-llama/Llama-3.3-70B-Instruct`

---

## Concepts<a id="concepts"></a>

### Agents<a id="agents"></a>

- They can have a custom model (e.g., `OpenAIModel`, a Llama-based model, etc.).
- Agents can have tools attached, which are specialized functions accessible during execution.
- Agents can have a custom Persona Profile configured.

### Tasks<a id="tasks"></a>

- A **task** is a single step in a workflow, e.g.,  `schedule_reminder`, `sentiment`, `translate_text`, etc.
- Tasks are managed by the `Workflow` class in `workflow.py`.
- Tasks can be chained for multi-step logic into a workflow (e.g., `await Workflow(objective="...").translate_text().sentiment().run_tasks()`).

### Client Mode vs Local Mode<a id="client-mode-vs-local-mode"></a>

- **Local Mode**: The system calls `run_agents(...)` directly in your local environment.  
- **Client Mode**: The system calls out to remote endpoints in a separate API.
  - In `client_mode=True`, each task (e.g. `sentiment`) triggers a client function (`sentiment_analysis(...)`) instead of local logic.

This allows you to **switch** between running tasks locally or delegating them to a server.

### Workflows (YAML/JSON)<a id="workflows-yamljson"></a>

_Note: this part is under active development and might not always function!_

- You can define multi-step workflows in YAML or JSON.
- The endpoint `/run-file` accepts a file (via multipart form data).
  - First tries parsing the payload as **JSON**.
  - If that fails, it tries parsing the payload as **YAML**.
- The file is validated against a `WorkflowDefinition` Pydantic model.
- Each step has a `type` (e.g., `"sentiment"`, `"custom"`) and optional parameters (like `agents`, `target_language`, etc.).

---

## Usage<a id="usage"></a>

### Creating Agents<a id="creating-agents"></a>

```python
from iointel import Agent

my_agent = Agent(
    name="MyAgent",
    instructions="You are a helpful agent.",
    # one can also pass custom model using pydantic_ai.models.openai.OpenAIModel
    # or pass args to OpenAIModel() as kwargs to Agent()
)
```

### Creating an Agent with a Persona<a id="creating-an-agent-with-a-persona"></a>

```python
from iointel import PersonaConfig, Agent


my_persona = PersonaConfig(
    name="Elandria the Arcane Scholar",
    age=164,
    role="an ancient elven mage",
    style="formal and slightly archaic",
    domain_knowledge=["arcane magic", "elven history", "ancient runes"],
    quirks="often references centuries-old events casually",
    bio="Once studied at the Grand Academy of Runic Arts",
    lore="Elves in this world can live up to 300 years",
    personality="calm, wise, but sometimes condescending",
    conversation_style="uses 'thee' and 'thou' occasionally",
    description="Tall, silver-haired, wearing intricate robes with arcane symbols",
    emotional_stability=0.85,
    friendliness=0.45,
    creativity=0.68,
    curiosity=0.95,
    formality=0.1,
    empathy=0.57,
    humor=0.99,
)

agent = Agent(
    name="ArcaneScholarAgent",
    instructions="You are an assistant specialized in arcane knowledge.",
    persona=my_persona
)

print(agent.instructions)
```

### Building a Workflow<a id="building-a-workflow"></a>

In Python code, you can create tasks by instantiating the Tasks class and chaining methods:


```python
from iointel import Workflow

tasks = Workflow(objective="This is the text to analyze", client_mode=False)
(
  tasks
    .sentiment(agents=[my_agent])
    .translate_text(target_language="french")   # a second step
)

results = await tasks.run_tasks()
print(results)
```
Because client_mode=False, everything runs locally.

### Running a Local Workflow<a id="running-a-local-workflow"></a>

```python
tasks = Workflow(objective="Breaking news: local sports team wins!", client_mode=False)
await tasks.summarize_text(max_words=50).run_tasks()
```

### Running a Remote Workflow (Client Mode)<a id="running-a-remote-workflow-client-mode"></a>

```python
tasks = Workflow(objective="Breaking news: local sports team wins!", client_mode=True)
await tasks.summarize_text(max_words=50).run_tasks()
```
Now, summarize_text calls the client function (e.g., summarize_task(...)) instead of local logic.

### Uploading YAML/JSON Workflows<a id="uploading-yamljson-workflows"></a>
_Note: this part is under active development and might not always function!_

	1.	Create a YAML or JSON file specifying workflow:

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

	2.	Upload via the /run-file endpoint (multipart file upload).
The server reads it as JSON or YAML and runs the tasks sequentially in local mode.

## Examples<a id="examples"></a>

### Simple Summarize Task<a id="simple-summarize-task"></a>

```python
tasks = Workflow("Breaking news: new Python release!", client_mode=False)
await tasks.summarize_text(max_words=30).run_tasks()
```

Returns a summarized result.

### Chainable Workflows<a id="chainable-workflows"></a>

```python
tasks = Workflow("Tech giant acquires startup for $2B", client_mode=False)
(tasks
   .translate_text(target_language="spanish")
   .sentiment()
)
await results = tasks.run_tasks()
```

	1.	Translate to Spanish,
	2.	Sentiment analysis.

### Custom Workflow<a id="custom-workflow"></a>
```python
tasks = Workflow("Analyze this special text", client_mode=False)
tasks.custom(
    name="my-unique-step",
    objective="Perform advanced analysis",
    instructions="Focus on entity extraction and sentiment",
    agents=[my_agent],
    **{"extra_context": "some_val"}
)
await results = tasks.run_tasks()
```

A "custom" task can reference a custom function in the CUSTOM_WORKFLOW_REGISTRY or fall back to a default behavior.

### Loading From a YAML File<a id="loading-from-a-yaml-file"></a>
_Note: this part is under active development and might not always function!_

```bash
curl -X POST "https://api.intelligence.io.solutions/api/v1/workflows/run-file" \
     -F "yaml_file=@path/to/workflow.yaml"
```

## API Endpoints<a id="api-endpoints"></a>

Please refer to (IO.net documentation)[https://docs.io.net/docs/exploring-ai-agents] to see particular endpoints and their documentation.

## License<a id="license"></a>
See the [LICENSE](https://github.com/ionet-official/iointel?tab=Apache-2.0-1-ov-file#readme) file for license rights and limitations (Apache 2.0).

# IOIntel: Agentic Tools with Beautiful UI

## Features
- **Agentic tool use**: Agents can call Python tools, return results, and chain reasoning.
- **Rich tool call visualization**: Tool calls and results are rendered as beautiful, gold-accented "pills" in both CLI (with [rich](https://github.com/Textualize/rich)) and Gradio UI.
- **Dynamic UI**: Agents can generate forms (textboxes, sliders, etc.) on the fly in the Gradio app.
- **Live CSS theming**: Agents can change the UI theme at runtime.
- **Jupyter compatible**: The Gradio UI can be launched in a notebook cell.

---

## Quickstart: CLI Usage

```python
from iointel import Agent, register_tool

@register_tool
def add(a: float, b: float) -> float:
    return a + b

agent = Agent(
    name="Solar",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
    api_key="sk-...",
    tools=[add],
    show_tool_calls=True,  # Pretty rich tool call output!
)

import asyncio
async def main():
    result = await agent.run("What is 2 + 2?", pretty=True)
    # Tool calls/results are shown in rich formatting!

asyncio.run(main())
```

### Multimodal Support

iointel supports multimodal inputs through various content types:

```python
from iointel import Agent, ImageUrl, BinaryContent, DocumentUrl, AudioUrl, VideoUrl

agent = Agent(
    name="VisionAgent",
    instructions="You are a helpful vision assistant.",
    model="openai/gpt-oss-120b",
    api_key="io-...",
)

# Images
result = await agent.run([
    "What's in this image?",
    ImageUrl(url="https://example.com/image.png")
])

# Local images with binary content
with open("local_image.png", "rb") as f:
    image_data = f.read()

result = await agent.run([
    "Describe this image",
    BinaryContent(data=image_data, media_type="image/png")
])

# Documents
result = await agent.run([
    "Summarize this document",
    DocumentUrl(url="https://example.com/document.pdf")
])

# Audio/Video (model dependent)
result = await agent.run([
    "Transcribe this audio",
    AudioUrl(url="https://example.com/audio.mp3")
])
```

**Supported Media Types:**
The specific media types supported depend on your LLM model provider:

- **Images**: PNG, JPEG, GIF, WebP
- **Documents**: PDF, TXT
- **Audio/Video**: MP3, MP4, WAV (varies by provider)

Check your model provider's documentation for specific format support and limitations.
![Screenshot 2025-06-02 at 5 46 15 PM](https://github.com/user-attachments/assets/b563a937-bb06-4856-9ff2-d3f1ddda5a1a)

![Screenshot 2025-06-02 at 5 46 55 PM](https://github.com/user-attachments/assets/c52ca18b-375a-4406-9a5f-02bac598a6cf)

---

## Quickstart: Gradio UI

> **Note**: To use the Gradio UI, install with UI dependencies: `pip install "iointel[ui]"`

```python
from iointel import Agent, register_tool

@register_tool
def get_weather(city: str) -> dict:
    return {"temp": 72, "condition": "Sunny"}

agent = Agent(
    name="GradioSolar",
    instructions="You are a helpful assistant.",
    model="openai/gpt-oss-120b",
    api_key="io-...",
    tools=[get_weather],
    show_tool_calls=True,
)

# Launch the beautiful Gradio Chat UI (works in Jupyter too!)
await agent.launch_chat_ui(interface_title="Iointel Gradio Solar")

# Or, for more control across different agents:
# from iointel.src.ui.io_gradio_ui import IOGradioUI
# ui = IOGradioUI(agent, interface_title="Iointel GradioSolar")
# await ui.launch(share=True)
```

![Screenshot 2025-06-02 at 5 44 49 PM](https://github.com/user-attachments/assets/1b6f834c-1ff3-4475-b581-c1b5233a099e)


- **Tool calls** are rendered as beautiful, gold-trimmed panels in the chat.
- **Dynamic UI**: If your agent/tool returns a UI spec, it will be rendered live.
- **Works in Jupyter**: Just run the above in a notebook cell!

---
