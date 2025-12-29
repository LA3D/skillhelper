# EvolvingSkill: Skills That Learn From Experience

> Design document for integrating ReasoningBank-style learning into Agent Skills

## Overview

**EvolvingSkill** extends the Agent Skills framework with memory-based learning, enabling skills to improve through both human-in-the-loop feedback and autonomous self-execution. This design draws from the [ReasoningBank paper](https://arxiv.org/html/2509.25140v1) which demonstrates that distilling reasoning strategies from both successes and failures significantly improves agent performance.

## Motivation

Current Agent Skills are static:
- Instructions are author-defined and never change
- Skills don't learn from usage patterns
- Mistakes are repeated across sessions
- User preferences aren't captured

ReasoningBank shows that agents with memory:
- Improve 8+ percentage points on web tasks
- Reduce steps to completion by 20-30%
- Learn from failures (not just successes)
- Exhibit emergent strategy evolution

**Goal**: Combine the structure of Skills with the learning capabilities of ReasoningBank.

## Core Concepts

### Memory Items

Each memory item captures distilled reasoning from experience:

```yaml
id: mem_001
title: "Verify file exists before modification"
description: "Always check file existence to avoid silent failures"
content: |
  When modifying files with skill tools:
  1. First use `os.path.exists()` to verify target exists
  2. If creating, check parent directory exists
  3. Handle missing file gracefully with clear error message

  Learned from: trace_042 (failure - file not found)
source: self_execution | human_feedback
outcome: success | failure | correction
timestamp: 2025-12-20T10:30:00Z
```

**Key insight from ReasoningBank**: Memory items should be "human-interpretable and machine-usable" — abstract enough to transfer across tasks, concrete enough to guide action.

### Directory Structure

```
evolving-skill/
├── SKILL.md                    # Static base instructions
├── tools.py                    # Tool functions
├── memory/                     # Learned strategies
│   ├── memories.yaml           # Memory items
│   └── [retrieval artifacts]   # Implementation-dependent
├── traces/                     # Execution history
│   ├── trace_001.json
│   └── trace_002.json
└── config.yaml                 # Learning configuration
```

## Two Learning Loops

### 1. Human-in-the-Loop Learning

Users provide feedback during skill usage:

| Feedback Type | Example | Extracted Memory |
|---------------|---------|------------------|
| **Correction** | "Use X instead of Y" | "When [context], prefer X over Y" |
| **Approval** | "Yes, that's right" | "This approach works for [pattern]" |
| **Refinement** | "Also consider Z" | "Additionally check Z when [context]" |
| **Rejection** | "This didn't work" | "Avoid [approach] because [reason]" |

**Solveit integration**: After skill tool execution, capture user's next message as potential feedback signal.

```python
def process_feedback(skill, trace, user_message):
    """Extract learning from user feedback."""

    # Use LLM to classify and extract
    extraction = llm_extract(
        prompt=FEEDBACK_EXTRACTION_PROMPT,
        trace=trace,
        feedback=user_message
    )

    if extraction.is_learning_signal:
        memory_item = MemoryItem(
            title=extraction.title,
            description=extraction.description,
            content=extraction.content,
            source="human_feedback",
            outcome=extraction.outcome_type
        )
        skill.memory.add(memory_item)
```

### 2. Self-Execution Learning (Autonomous)

Skills learn from their own tool execution:

```
Execute skill tools → Observe outcome → Judge success/failure → Extract memory
```

**Outcome signals**:
- Test results (pass/fail)
- Error messages
- Task completion indicators
- Output validation

**Memory extraction prompts** (adapted from ReasoningBank appendix):

**For Success**:
```
You are analyzing a successful skill execution trace.

Trace:
{trace}

Task: Extract a transferable reasoning strategy that made this succeed.
Focus on:
- Key decision points that led to success
- Patterns that could apply to similar tasks
- Insights that aren't obvious from the skill instructions alone

Output a memory item with title, description, and content.
```

**For Failure**:
```
You are analyzing a failed skill execution trace.

Trace:
{trace}
Error: {error}

Task: Reflect on the causes of failure and derive preventive strategies.
Focus on:
- What went wrong and why
- Warning signs that could have predicted the failure
- How to avoid this failure in future similar tasks

Output a memory item with title, description, and content.
```

## Memory-Augmented Skill Activation

When activated, an EvolvingSkill augments its base instructions with relevant memories:

```python
def to_prompt(self) -> str:
    """Generate prompt with dynamic memory augmentation."""
    base = self.instructions

    # Retrieve relevant memories (implementation TBD - see Research Questions)
    memories = self.memory.retrieve(context=current_context, k=3)

    if memories:
        augmented = f"""{base}

## Learned Strategies

The following insights were learned from past experience with this skill:

"""
        for m in memories:
            augmented += f"### {m.title}\n{m.content}\n\n"

        return augmented

    return base
```

## Emergent Strategy Evolution

ReasoningBank observed that strategies evolve through stages:

| Stage | Characteristics | Example |
|-------|-----------------|---------|
| **Procedural** | Basic tool execution | "Run linter on file" |
| **Adaptive** | Self-correction patterns | "If linter fails, check encoding first" |
| **Compositional** | Strategic reasoning | "Cross-reference style guide with project conventions" |
| **Meta-strategic** | Domain wisdom | "This codebase prefers explicit over implicit" |

This evolution happens naturally as memories accumulate and consolidate.

## Memory-Aware Test-Time Scaling (MaTTS)

For complex tasks, generate multiple attempts and learn from the contrast:

```python
def skill_with_matts(skill, task, k=3):
    """Generate k attempts, learn from success/failure contrast."""

    traces = []
    for i in range(k):
        trace = skill.execute(task, temperature=0.7 + i*0.1)
        traces.append(trace)

    successes = [t for t in traces if t.succeeded]
    failures = [t for t in traces if not t.succeeded]

    # Key insight: Contrastive learning from same task
    if successes and failures:
        memory = extract_contrastive_insight(successes, failures)
        skill.memory.add(memory)

    return select_best(traces)
```

**Why this works**: Multiple attempts on the same task provide controlled comparison — same goal, different approaches, different outcomes.

## Configuration

```yaml
# config.yaml
learning:
  enabled: true

  human_feedback:
    enabled: true
    auto_detect: true  # Try to detect feedback signals

  self_execution:
    enabled: true
    extract_on_success: true
    extract_on_failure: true

  consolidation:
    enabled: true
    interval: "weekly"  # Merge similar memories
    max_memories: 100

retrieval:
  method: "tbd"  # See Research Questions
  k: 3  # Number of memories to retrieve
  include_failures: true
```

---

## Research Questions

### Memory Retrieval

**Problem**: Given a current task context, how do we retrieve the most relevant memories?

This is an open research question. We should explore options from simplest to most complex:

#### Option 1: No Retrieval (Always Include All)

```python
def retrieve(self, k=None):
    return self.memories[-k:] if k else self.memories
```

- **Pros**: Simplest possible; no infrastructure needed
- **Cons**: Doesn't scale; irrelevant memories waste context
- **When to use**: Small memory banks (<20 items), early prototyping

#### Option 2: Recency-Based

```python
def retrieve(self, k=3):
    return sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:k]
```

- **Pros**: Simple; recent memories often most relevant
- **Cons**: May miss important older memories
- **When to use**: Tasks with temporal locality

#### Option 3: Keyword/Tag Matching

```python
def retrieve(self, context, k=3):
    scored = []
    context_words = set(context.lower().split())
    for m in self.memories:
        memory_words = set(m.content.lower().split())
        overlap = len(context_words & memory_words)
        scored.append((overlap, m))
    return [m for _, m in sorted(scored, reverse=True)[:k]]
```

- **Pros**: No embeddings needed; interpretable
- **Cons**: Misses semantic similarity
- **When to use**: When embedding infrastructure unavailable

#### Option 4: Embedding-Based Semantic Search

```python
def retrieve(self, context, k=3):
    context_embedding = embed(context)
    scored = []
    for m in self.memories:
        if m.embedding is None:
            m.embedding = embed(m.content)
        score = cosine_similarity(context_embedding, m.embedding)
        scored.append((score, m))
    return [m for _, m in sorted(scored, reverse=True)[:k]]
```

- **Pros**: Captures semantic similarity
- **Cons**: Requires embedding model; adds latency/cost
- **When to use**: Large memory banks; diverse task types

#### Option 5: LLM-Based Selection

```python
def retrieve(self, context, k=3):
    prompt = f"""
Given this task context:
{context}

Select the {k} most relevant memories from:
{format_memories(self.memories)}

Return the IDs of the most relevant memories.
"""
    selected_ids = llm(prompt)
    return [m for m in self.memories if m.id in selected_ids]
```

- **Pros**: Most flexible; can reason about relevance
- **Cons**: Expensive; adds latency; may be inconsistent
- **When to use**: High-stakes tasks; complex relevance judgments

#### Hybrid Approaches

- **Tiered**: Keyword filter → Embedding rerank
- **Ensemble**: Average scores from multiple methods
- **Adaptive**: Choose method based on memory bank size

#### Evaluation Questions

1. **Accuracy**: Does retrieval return memories that actually help?
2. **Efficiency**: What's the latency/cost overhead?
3. **Robustness**: Does it work across different task types?
4. **Simplicity**: Is the added complexity justified?

**Recommendation**: Start with Option 1 or 2 for prototyping. Measure whether retrieval quality is actually a bottleneck before adding complexity.

---

### Memory Consolidation

**Problem**: How do we merge, prune, or evolve memories over time?

Questions to explore:
- When are two memories similar enough to merge?
- How do we handle contradictory memories?
- Should older memories decay or be pruned?
- Can memories be hierarchically organized?

### Memory Privacy

**Problem**: When skills are shared/distributed, what happens to learned memories?

Options:
- **Strip memories**: Distribute skill without learned memories
- **Anonymize**: Remove user-specific information
- **Curate**: Author selects which memories to include
- **Federated**: Learn patterns without sharing raw data

### Cold Start

**Problem**: New skills have no memories. How do we bootstrap?

Options:
- **Synthetic generation**: LLM generates likely useful strategies
- **Transfer**: Copy relevant memories from similar skills
- **Author seeding**: Skill author provides initial memories
- **Accelerated learning**: Aggressive MaTTS on first uses

### Verification

**Problem**: How do we ensure learned strategies don't introduce errors?

Options:
- **Confidence scoring**: Track success rate of memory-guided actions
- **A/B testing**: Compare with/without memory randomly
- **Human review**: Flag memories for periodic review
- **Rollback**: Maintain memory checkpoints

---

## Implementation Phases

### Phase 1: Memory Storage

- [ ] Define `MemoryItem` dataclass
- [ ] Implement `SkillMemory` class (YAML-based storage)
- [ ] Add `memory/` directory to skill structure
- [ ] Basic memory add/list/delete operations

### Phase 2: Trace Recording

- [ ] Define `ExecutionTrace` dataclass
- [ ] Instrument skill tool execution to record traces
- [ ] Implement `TraceStore` class
- [ ] Add outcome detection (success/failure signals)

### Phase 3: Memory Extraction

- [ ] Implement extraction prompts (success/failure variants)
- [ ] Add human feedback detection in solveit
- [ ] Create `extract_memory()` function
- [ ] Trigger extraction after tool execution

### Phase 4: Memory-Augmented Activation

- [ ] Modify `to_prompt()` to include memories
- [ ] Implement simplest retrieval (recency-based)
- [ ] Test with manual memory creation
- [ ] Measure impact on task success

### Phase 5: Research & Iteration

- [ ] Experiment with retrieval methods
- [ ] Implement memory consolidation
- [ ] Add MaTTS for test-time scaling
- [ ] Evaluate emergent evolution

---

## Building EvolvingSkills with Solveit

This section describes how to implement EvolvingSkills using the [solveit](https://www.answer.ai/posts/2025-10-01-solveit-full.html) platform and its design philosophy.

### Solveit Philosophy Alignment

Solveit is built on **George Polya's "How to Solve It"** (1945):
1. **Understand** the problem
2. **Devise** a plan
3. **Execute** the steps
4. **Reflect** on results

This maps directly to EvolvingSkill learning:

| Polya Step | Solveit Action | EvolvingSkill Learning |
|------------|----------------|------------------------|
| Understand | Read context, explore | Retrieve relevant memories |
| Devise | Plan approach | Augment instructions with strategies |
| Execute | Run skill tools | Record execution trace |
| Reflect | Edit, refine, iterate | Extract and store memories |

**Key principle**: "Collaboration not replacement" — the human remains in control. EvolvingSkills learn *from* human guidance, not *instead of* it.

### Dialog Engineering as Learning Signals

Solveit treats dialogs as **"living documents you refine over time rather than append-only logs."** Every refinement is a potential learning signal:

| Dialog Action | What It Signals | Memory Extraction |
|---------------|-----------------|-------------------|
| **Edit AI response** | Correction needed | "Prefer [edited version] over [original]" |
| **Delete message** | Dead end, wrong approach | "Avoid [approach] in [context]" |
| **Pin message** | Important context | "Always consider [pinned info]" |
| **Re-run with changes** | Refinement | "When [condition], also [addition]" |
| **Accept and continue** | Approval | "This approach works for [pattern]" |

#### Detecting Dialog Engineering Events

Using `dialoghelper`, we can detect these signals:

```python
#| export
from dialoghelper import find_msgs, read_msg, curr_dialog

def detect_dialog_edits(skill_activation_msg_id: str) -> list[dict]:
    """Detect edits made after skill activation.

    Returns list of edit events with before/after content.
    """
    events = []
    dialog = curr_dialog(with_messages=True)

    # Find messages after skill activation
    activation_idx = None
    for i, msg in enumerate(dialog['messages']):
        if msg.get('id') == skill_activation_msg_id:
            activation_idx = i
            break

    if activation_idx is None:
        return events

    # Check subsequent messages for edit markers
    for msg in dialog['messages'][activation_idx:]:
        if msg.get('edited'):
            events.append({
                'type': 'edit',
                'msg_id': msg['id'],
                'original': msg.get('original_content'),
                'edited': msg.get('content'),
                'msg_type': msg.get('msg_type')
            })
        if msg.get('deleted'):
            events.append({
                'type': 'delete',
                'msg_id': msg['id'],
                'content': msg.get('content'),
                'msg_type': msg.get('msg_type')
            })

    return events
```

### The Three Message Types

Solveit has three message types, each relevant to skill learning:

```
┌─────────────────────────────────────────────────────────────┐
│  CODE MESSAGE                                               │
│  - Execute Python in persistent interpreter                 │
│  - Output displayed below                                   │
│  - LEARNING: Execution traces, success/failure signals      │
├─────────────────────────────────────────────────────────────┤
│  NOTE MESSAGE                                               │
│  - Markdown documentation                                   │
│  - Human context and reasoning                              │
│  - LEARNING: Skill instructions injected here               │
├─────────────────────────────────────────────────────────────┤
│  PROMPT MESSAGE                                             │
│  - User queries to AI                                       │
│  - AI responses with tool calls                             │
│  - LEARNING: Human feedback, corrections, approvals         │
└─────────────────────────────────────────────────────────────┘
```

### Skill Integration Pattern

Here's how an EvolvingSkill integrates with a solveit dialog:

#### Step 1: Activate Skill (Code Message)

```python
# In a code cell
from skillhelper.skilleddialog import use_skill

result = use_skill('code-reviewer')
globals().update(result.tools_dict)

# Store activation context for learning
_skill_context = {
    'skill': result.skilled_dialog,
    'activation_msg_id': find_msg_id(),
    'trace': []
}
```

#### Step 2: Skill Instructions Injected (Note Message)

The skill's instructions (augmented with memories) appear as a note:

```markdown
[Skill activated: code-reviewer]

# Code Reviewer Skill

This skill helps review code for common issues...

## Learned Strategies

### Verify file encoding before linting
When reviewing files, first check encoding to avoid false positives
from encoding-related parse errors.

### User prefers severity ratings
Include severity (critical/warning/info) for each issue found.
```

#### Step 3: Use Tools (Prompt Message)

```markdown
Please review this code for issues:

```python
def process(data):
    return data.split(',')
```

& `[check_style, count_lines]`
```

#### Step 4: Record Trace (Automatic)

```python
# Wrapper that records tool execution
def traced_tool(func, skill_context):
    """Wrap skill tool to record execution trace."""
    def wrapper(*args, **kwargs):
        call_record = {
            'tool': func.__name__,
            'args': args,
            'kwargs': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        try:
            result = func(*args, **kwargs)
            call_record['result'] = str(result)[:500]  # Truncate
            call_record['success'] = True
        except Exception as e:
            call_record['error'] = str(e)
            call_record['success'] = False
            raise
        finally:
            skill_context['trace'].append(call_record)
        return result
    return wrapper
```

#### Step 5: Detect Feedback & Learn (After Interaction)

```python
# In a code cell, after the skill interaction
from skillhelper.evolvingskill import learn_from_session

# Analyze the dialog for learning signals
memories = learn_from_session(
    skill=_skill_context['skill'],
    activation_msg_id=_skill_context['activation_msg_id'],
    trace=_skill_context['trace']
)

if memories:
    print(f"Learned {len(memories)} new strategies")
    for m in memories:
        print(f"  - {m.title}")
```

### Solveit-Specific Feedback Signals

Beyond explicit feedback, solveit's workflow provides implicit signals:

#### 1. Code Execution Results

```python
def detect_execution_outcome(trace: list) -> dict:
    """Analyze code execution for success/failure signals."""

    signals = {
        'test_results': None,
        'exceptions': [],
        'outputs': []
    }

    for call in trace:
        if not call.get('success'):
            signals['exceptions'].append(call.get('error'))

        result = call.get('result', '')

        # Detect test patterns
        if 'PASSED' in result or 'OK' in result:
            signals['test_results'] = 'pass'
        elif 'FAILED' in result or 'ERROR' in result:
            signals['test_results'] = 'fail'

        signals['outputs'].append(result)

    return signals
```

#### 2. User's Next Action

The user's next action after AI response is highly informative:

| Next Action | Signal | Learning |
|-------------|--------|----------|
| Runs suggested code | Tentative approval | Weak positive |
| Runs and continues | Strong approval | Record success pattern |
| Edits before running | Correction | Extract difference |
| Deletes and re-prompts | Rejection | Record failure |
| Adds note with context | Refinement | Augment strategy |

```python
def analyze_user_response(activation_idx: int, dialog: dict) -> dict:
    """Analyze user's response to skill output."""

    messages = dialog['messages']
    ai_response_idx = activation_idx + 1  # Assuming immediate response

    if ai_response_idx + 1 >= len(messages):
        return {'type': 'no_response'}

    next_msg = messages[ai_response_idx + 1]

    if next_msg['msg_type'] == 'code':
        # User ran code - check if it's the suggested code
        ai_response = messages[ai_response_idx]
        suggested_code = extract_code_blocks(ai_response['content'])
        user_code = next_msg['content']

        if user_code in suggested_code:
            return {'type': 'approval', 'strength': 'strong'}
        else:
            return {'type': 'modification', 'original': suggested_code, 'modified': user_code}

    elif next_msg['msg_type'] == 'prompt':
        # User sent another prompt
        if is_correction(next_msg['content']):
            return {'type': 'correction', 'feedback': next_msg['content']}
        else:
            return {'type': 'continuation'}

    elif next_msg['msg_type'] == 'note':
        return {'type': 'documentation', 'content': next_msg['content']}

    return {'type': 'unknown'}
```

### Memory Storage in Solveit Context

Where do memories live in solveit?

#### Option A: Skill Directory (Portable)

```
~/.skills/code-reviewer/
├── SKILL.md
├── tools.py
└── memory/
    └── memories.yaml   # Travels with the skill
```

**Pros**: Portable, skill-specific
**Cons**: Not project-aware

#### Option B: Project Directory (Context-Aware)

```
my-project/
├── .solveit/
│   └── skill_memories/
│       └── code-reviewer.yaml   # Project-specific learning
├── src/
└── tests/
```

**Pros**: Project-specific patterns
**Cons**: Doesn't transfer to other projects

#### Option C: Hybrid (Recommended)

```python
def get_memory_path(skill_name: str) -> Path:
    """Get memory path with project override."""

    # Check for project-local memories
    project_path = Path('.solveit/skill_memories') / f'{skill_name}.yaml'
    if project_path.exists():
        return project_path

    # Fall back to skill's own memory
    skill = find_skill(skill_name)
    return skill.path / 'memory' / 'memories.yaml'

def save_memory(skill_name: str, memory: MemoryItem, scope: str = 'project'):
    """Save memory to appropriate location."""

    if scope == 'project':
        path = Path('.solveit/skill_memories')
        path.mkdir(parents=True, exist_ok=True)
        memory_file = path / f'{skill_name}.yaml'
    else:  # scope == 'skill'
        skill = find_skill(skill_name)
        memory_file = skill.path / 'memory' / 'memories.yaml'

    # Append to existing memories
    existing = load_memories(memory_file) if memory_file.exists() else []
    existing.append(memory)
    save_memories(memory_file, existing)
```

### Complete Solveit Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER ACTIVATES SKILL                                         │
│    result = use_skill('code-reviewer')                          │
│    globals().update(result.tools_dict)                          │
├─────────────────────────────────────────────────────────────────┤
│ 2. SKILL INSTRUCTIONS INJECTED (with memories)                  │
│    [Note message with augmented instructions]                   │
├─────────────────────────────────────────────────────────────────┤
│ 3. USER PROMPTS WITH TOOLS                                      │
│    "Review this code..." & `[check_style]`                      │
├─────────────────────────────────────────────────────────────────┤
│ 4. AI EXECUTES TOOLS, RESPONDS                                  │
│    [Tool calls recorded in trace]                               │
├─────────────────────────────────────────────────────────────────┤
│ 5. USER REACTS                                                  │
│    - Runs code → approval                                       │
│    - Edits response → correction                                │
│    - Deletes → rejection                                        │
│    - Continues → implicit approval                              │
├─────────────────────────────────────────────────────────────────┤
│ 6. LEARNING EXTRACTED                                           │
│    memories = learn_from_session(...)                           │
│    [Saved to project or skill memory]                           │
├─────────────────────────────────────────────────────────────────┤
│ 7. NEXT SESSION                                                 │
│    [Memories retrieved and augment instructions]                │
└─────────────────────────────────────────────────────────────────┘
```

### Solveit-Native Implementation

Here's a minimal implementation using solveit conventions:

```python
# evolvingskill.py - Solveit-native EvolvingSkill

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class MemoryItem:
    """A learned strategy from experience."""
    id: str
    title: str
    description: str
    content: str
    source: str  # 'human_feedback' | 'self_execution' | 'dialog_edit'
    outcome: str  # 'success' | 'failure' | 'correction'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SkillMemory:
    """Simple YAML-based memory storage."""

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.memories_file = self.path / 'memories.yaml'

    def load(self) -> list[MemoryItem]:
        if not self.memories_file.exists():
            return []
        with open(self.memories_file) as f:
            data = yaml.safe_load(f) or []
        return [MemoryItem(**m) for m in data]

    def save(self, memories: list[MemoryItem]):
        with open(self.memories_file, 'w') as f:
            yaml.dump([m.__dict__ for m in memories], f)

    def add(self, memory: MemoryItem):
        memories = self.load()
        memories.append(memory)
        self.save(memories)

    def retrieve(self, k: int = 3) -> list[MemoryItem]:
        """Retrieve most recent memories (simplest retrieval)."""
        memories = self.load()
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)[:k]


def use_evolving_skill(name: str, inject_msg: bool = True):
    """Activate an evolving skill with memory augmentation.

    Usage in solveit:
        result = use_evolving_skill('code-reviewer')
        globals().update(result.tools_dict)
        print(result.declaration)
    """
    from skillhelper.skilleddialog import SkilledDialog
    from skillhelper.core import discover_all
    from fastcore.basics import first, AttrDict

    # Find skill
    skill = first(s for s in discover_all() if s.name == name)
    if not skill:
        return AttrDict(success=False, error=f"Skill '{name}' not found")

    # Load memories
    memory = SkillMemory(skill.path / 'memory')
    memories = memory.retrieve(k=3)

    # Augment instructions with memories
    instructions = skill.instructions
    if memories:
        instructions += "\n\n## Learned Strategies\n\n"
        instructions += "The following were learned from past experience:\n\n"
        for m in memories:
            instructions += f"### {m.title}\n{m.content}\n\n"

    # Create skilled dialog with augmented instructions
    sd = SkilledDialog()
    result = sd.activate_skill(name, inject_msg=False)

    # Inject augmented instructions
    if inject_msg:
        try:
            from dialoghelper import add_msg, find_msg_id
            msg_content = f"[Skill activated: {name}]\n\n{instructions}\n\n"
            msg_content += f"**Tools available:** {result.declaration}"
            add_msg(msg_content, msg_type='note')
            result.activation_msg_id = find_msg_id()
        except ImportError:
            pass  # Not in solveit

    result.memory = memory
    result.augmented_instructions = instructions
    return result


def learn_from_feedback(skill_name: str, feedback: str, trace: list = None):
    """Manually trigger learning from user feedback.

    Usage in solveit (after skill interaction):
        learn_from_feedback('code-reviewer', 'Always check for None first')
    """
    from skillhelper.core import discover_all
    from fastcore.basics import first
    import uuid

    skill = first(s for s in discover_all() if s.name == skill_name)
    if not skill:
        print(f"Skill '{skill_name}' not found")
        return

    memory = SkillMemory(skill.path / 'memory')

    # Create memory item from feedback
    item = MemoryItem(
        id=f"mem_{uuid.uuid4().hex[:8]}",
        title=feedback[:50] + "..." if len(feedback) > 50 else feedback,
        description=feedback,
        content=feedback,
        source='human_feedback',
        outcome='correction'
    )

    memory.add(item)
    print(f"Learned: {item.title}")
```

### Usage Example in Solveit Dialog

```python
# === Code Cell 1: Activate skill ===
from skillhelper.evolvingskill import use_evolving_skill, learn_from_feedback

result = use_evolving_skill('code-reviewer')
globals().update(result.tools_dict)
print(result.declaration)
```

```markdown
# === Note Cell: Instructions appear here ===
[Skill activated: code-reviewer]

# Code Reviewer Skill
...

## Learned Strategies

### Always check for None before string operations
When reviewing code that processes strings, verify None handling...
```

```markdown
# === Prompt Cell: User asks for review ===
Review this function:

```python
def get_name(user):
    return user['name'].upper()
```

& `[check_style]`
```

```python
# === Code Cell 2: After interaction, teach the skill ===
# If the AI missed something important, teach it:
learn_from_feedback('code-reviewer',
    'Check for KeyError when accessing dict keys - use .get() or check "in"')
```

---

## References

1. [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/html/2509.25140v1) - Core concepts for memory extraction and MaTTS
2. [Anthropic Memory Cookbook](./dialogs/memory_cookbook.ipynb) - Memory tool implementation patterns
3. [Agent Skills Specification](https://agentskills.io/specification) - Base skill structure
4. [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - Context management strategies
5. [Launching Solveit](https://www.answer.ai/posts/2025-10-01-solveit-full.html) - Solveit philosophy and design
6. [Solveit Features Guide](https://www.fast.ai/posts/2025-11-07-solveit-features.html) - Technical details on dialogs and tools

---

## Appendix: Memory Extraction Prompts

### Success Extraction (Full)

```
You are analyzing a successful execution of the "{skill_name}" skill.

## Skill Description
{skill_description}

## Execution Trace
{trace}

## Outcome
Success: {success_indicator}

## Task
Extract a transferable reasoning strategy from this successful execution.

Focus on:
1. Key decision points that led to success
2. Patterns that could apply to similar future tasks
3. Insights not obvious from the base skill instructions
4. Specific techniques or approaches that worked well

## Output Format
Provide a memory item in this format:

Title: [Concise identifier, 5-10 words]
Description: [One sentence summary]
Content: [Detailed strategy, 3-5 bullet points or short paragraphs]
```

### Failure Extraction (Full)

```
You are analyzing a failed execution of the "{skill_name}" skill.

## Skill Description
{skill_description}

## Execution Trace
{trace}

## Error/Failure
{error}

## Task
Reflect on this failure and extract preventive strategies.

Focus on:
1. Root cause of the failure
2. Warning signs that could have predicted it
3. How to detect and avoid this failure pattern
4. Alternative approaches that might have succeeded

## Output Format
Provide a memory item in this format:

Title: [Concise identifier, 5-10 words]
Description: [One sentence summary]
Content: [Preventive strategy, 3-5 bullet points or short paragraphs]
```

### Human Feedback Extraction

```
You are analyzing user feedback after a "{skill_name}" skill execution.

## Execution Trace
{trace}

## User Feedback
{user_message}

## Task
Determine if this feedback contains a learning signal, and if so, extract it.

Feedback types:
- Correction: User indicates the approach was wrong
- Refinement: User adds additional considerations
- Approval: User confirms the approach was correct
- Rejection: User indicates complete failure

If this is NOT a learning signal (e.g., just a follow-up question), respond with:
NOT_A_LEARNING_SIGNAL

Otherwise, provide a memory item:

Title: [Concise identifier]
Description: [One sentence]
Content: [What was learned from this feedback]
Feedback_Type: [correction|refinement|approval|rejection]
```
