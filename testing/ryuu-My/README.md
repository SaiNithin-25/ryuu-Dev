# Ryuu-My

Ryuu-My is the personal intelligence layer of the Ryuu ecosystem.

While other Ryuu models focus on specialized work such as coding, research, gaming, or creativity, Ryuu-My is designed for everyday user support. It acts like a personal and voice assistant that can manage tasks, remember context, help with productivity, and serve as the user's daily conversational interface to the wider Ryuu-AI system.

## Current MVP

The repository now includes a working first build of Ryuu-My:

- A keyword-based intent classifier
- A task interpreter for reminders, tasks, notes, and schedule items
- A SQLite-backed memory store
- A simple local CLI in `cli.py`
- Seed training data in `data/seed/ryuu_my_seed.jsonl`
- Unit tests for the first assistant flow

## Quick Start

Run the local assistant:

```bash
python cli.py
```

Example prompts:

- `remind me to train ryuu tomorrow at 9am`
- `schedule meeting with the team monday at 3pm`
- `save this build a voice-first planner`
- `what is on my schedule`

## Core Role

Ryuu-My is responsible for user-centric assistance, including:

- Personal management
- Task lists
- Reminders
- Calendar scheduling
- Notes
- Project tracking
- Conversational interaction
- Friendly chat
- Context awareness
- Personality-driven responses
- Productivity assistance
- Email drafting
- Meeting summaries
- Planning workflows
- Device and system commands
- File organization
- Local automation
- Voice interaction
- Speech-to-text
- Intent recognition
- Text-to-speech

## Position in the Ryuu Ecosystem

Ryuu-My operates under the control of Ryuu-Modus, which routes user requests to the correct subsystem.

Example flow:

```text
User: remind me to train the model tomorrow at 9

Ryuu-Modus
  -> intent detected: personal task
  -> route to Ryuu-My
  -> create reminder
  -> store in memory
```

## High-Level Architecture

The main Ryuu-My pipeline is:

```text
User
  -> Speech Recognition (optional)
  -> Intent Detection
  -> Task Interpreter
  -> Memory System
  -> Task Engine
  -> Response Generator
```

## Internal Modules

### 1. Intent Classifier

Identifies the type of user request.

Example intents:

- `reminder` -> "remind me tomorrow"
- `schedule` -> "schedule meeting"
- `note` -> "save this idea"
- `task` -> "add todo"
- `question` -> "what is on my schedule"

### 2. Task Interpreter

Transforms natural language into structured commands that the system can store and execute.

Example:

User input:

```text
remind me to train ryuu tomorrow at 9am
```

Structured output:

```json
{
  "task": "reminder",
  "title": "Train Ryuu model",
  "time": "2026-03-11T09:00",
  "priority": "normal"
}
```

### 3. Memory System

Ryuu-My depends heavily on memory.

Short-term memory stores recent conversation context.

Example:

```text
User: schedule meeting tomorrow
User: make it 3pm
```

Long-term memory stores:

- User preferences
- Schedule
- Projects
- Habits
- Reminders

Example structure:

```json
{
  "user_preferences": {
    "wake_time": "7:00",
    "coding_time": "night"
  },
  "projects": [
    "Ryuu AI",
    "Game Engine"
  ]
}
```

## Voice Pipeline

If voice features are enabled, the interaction flow looks like this:

```text
User speech
  -> ASR (speech to text)
  -> Ryuu-My
  -> TTS (text to speech)
  -> Audio response
```

Possible tools:

- Speech recognition: Whisper
- Text-to-speech: Coqui or ElevenLabs
- Voice intent handling: Ryuu-My

## Training Data Direction

Good dataset categories for Ryuu-My include:

- Personal assistant datasets
- Task-oriented dialogue datasets
- Scheduling datasets
- Calendar data
- Chit-chat datasets
- Friendly conversation logs
- Command datasets

Example data format:

```json
{
  "prompt": "add milk to my grocery list",
  "completion": "Item added to grocery list."
}
```

### Recommended Data Types

1. Command understanding
2. Conversational interaction
3. Productivity actions

Examples:

- "remind me tomorrow"
- "schedule meeting"
- "add to list"
- "how was your day"
- "tell me something interesting"
- "write email"
- "summarize notes"
- "plan tasks"

## Suggested Model Configuration

Ryuu-My does not need to be extremely large. A lightweight assistant model is enough for many local assistant workflows.

Suggested range:

- Layers: 8-10
- Heads: 8
- Embedding size: 512
- Context length: 512
- Total parameters: about 50M-120M

## Example Workflow

User request:

```text
what should I work on today
```

Pipeline:

```text
Ryuu-Modus
  -> Ryuu-My
  -> check memory database
  -> inspect tasks
  -> generate suggestion
```

Possible response:

```text
You planned to work on:
- Ryuu-Dev dataset
- Research dataset collection
- Model training improvements
```

## Data Storage

Ryuu-My data can be stored in SQLite or JSON-backed storage.

Suggested database:

```text
user_data.db
```

Suggested tables:

- tasks
- notes
- preferences
- reminders
- projects

## Integration With Other Ryuu Models

Ryuu-My can delegate expert tasks to other subsystems and then present the result conversationally.

Example:

```text
User: explain transformers to me

Ryuu-Modus
  -> Ryuu-Re
  -> explanation returned
  -> Ryuu-My presents it in a user-friendly style
```

## Example Use Cases

- Daily assistant: "what's my schedule today"
- Project manager: "add task to train ryuu model"
- Learning companion: "help me understand neural networks"
- System automation: "open my training script"

## Future Features

Ryuu-My can later expand into:

- Habit tracking
- Life analytics
- Emotional AI
- Voice conversation
- Local device control
- Smart home integration

## Role Map

| Model | Role |
| --- | --- |
| Ryuu-Dev | Developer expert |
| Ryuu-Re | Research expert |
| Ryuu-GenArt | Creative expert |
| Ryuu-G | Game expert |
| Ryuu-My | Personal assistant |
| Ryuu-Modus | Orchestrator |

Ryuu-My acts as the user's daily interface to the entire Ryuu ecosystem.

