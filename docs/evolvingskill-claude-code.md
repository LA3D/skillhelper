# EvolvingSkills with Claude Code

> How Claude Code can use skills that learn from experience

## Overview

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is Anthropic's CLI tool that supports Agent Skills via the SKILL.md specification. This document describes how EvolvingSkills—skills that learn from experience—can work within Claude Code's architecture.

## How Claude Code Loads Skills

Claude Code's skill activation flow:

```
1. User installs skill to ~/.claude/skills/ or project .claude/skills/
2. Skill metadata (name, description) loaded into system prompt
3. When skill is triggered, Claude uses bash to read SKILL.md
4. If instructions reference other files, Claude reads those too
5. Claude executes the skill's guidance
```

Key insight: **Claude Code already reads files from skill directories.** We can leverage this for memory integration.

## Integration Strategies

### Strategy 1: Self-Referential Instructions (Recommended)

The simplest approach: tell Claude to check for memories in the skill instructions.

**SKILL.md:**
```markdown
---
name: code-reviewer
description: Reviews code for quality issues. Learns from experience.
---

# Code Reviewer Skill

## Before Starting

This skill learns from experience. Before reviewing code:

1. Check if `memory/learned-strategies.md` exists in this skill directory
2. If it exists, read it to see patterns learned from past reviews
3. Apply these learned strategies alongside the base instructions

## Core Instructions

When reviewing code, analyze for:
- Style violations and formatting issues
- Potential bugs and edge cases
- Performance concerns
- Security vulnerabilities

## After Completing a Review

If you discovered a useful pattern or insight worth remembering:

1. Read the current `memory/learned-strategies.md` (create if needed)
2. Append a new entry with:
   - **Title**: Brief identifier (5-10 words)
   - **Context**: When this applies
   - **Strategy**: What to do
   - **Learned from**: Brief description of the experience

Example entry:
```
### Check encoding before parsing
**Context**: When reviewing file processing code
**Strategy**: Verify file encoding handling before analyzing parse logic;
encoding issues often masquerade as parsing bugs.
**Learned from**: Spent 20 minutes debugging a "parser bug" that was UTF-8 BOM
```
```

**Directory Structure:**
```
~/.claude/skills/code-reviewer/
├── SKILL.md
├── tools.py                      # Optional
├── references/
│   └── style-guide.md
└── memory/
    └── learned-strategies.md     # Claude reads and writes this
```

**Pros:**
- No external tooling required
- Works immediately with current Claude Code
- Human-readable memory format
- Claude naturally follows file-based instructions

**Cons:**
- Relies on Claude following instructions consistently
- No structured validation of memory entries
- Memory format is freeform

---

### Strategy 2: References Pattern

Use the existing `references/` directory that Claude Code already understands.

**SKILL.md:**
```markdown
---
name: code-reviewer
description: Reviews code with accumulated expertise
---

# Code Reviewer Skill

## References

Load these references before starting:
- `references/base-checklist.md` - Core review criteria
- `references/learned-patterns.md` - Patterns from past reviews (if exists)

## Instructions
...
```

A background process syncs structured memories to the reference file:

```python
# sync_memories.py - Run periodically or via hook
import yaml
from pathlib import Path

def sync_memories_to_reference(skill_path: Path):
    """Convert structured memories to readable reference doc."""

    memories_file = skill_path / 'memory' / 'memories.yaml'
    reference_file = skill_path / 'references' / 'learned-patterns.md'

    if not memories_file.exists():
        return

    with open(memories_file) as f:
        memories = yaml.safe_load(f) or []

    # Generate markdown reference
    content = "# Learned Patterns\n\n"
    content += "These patterns were learned from past experience.\n\n"

    for m in memories:
        content += f"## {m['title']}\n\n"
        content += f"{m['content']}\n\n"
        content += f"*Source: {m['source']} | Outcome: {m['outcome']}*\n\n"
        content += "---\n\n"

    reference_file.parent.mkdir(parents=True, exist_ok=True)
    with open(reference_file, 'w') as f:
        f.write(content)
```

**Pros:**
- Structured storage (YAML) with readable output (Markdown)
- Fits Claude Code's existing reference pattern
- Can filter/curate memories before exposing

**Cons:**
- Requires sync step
- Two sources of truth (memories.yaml vs learned-patterns.md)

---

### Strategy 3: MCP Memory Server

Build memory management as an [MCP server](https://modelcontextprotocol.io/) that Claude Code can use.

**MCP Server Definition:**
```python
# skill_memory_server.py
from mcp import Server
import yaml
from pathlib import Path

server = Server("skill-memory")

@server.tool()
def get_skill_memories(skill_name: str, k: int = 5) -> str:
    """Retrieve learned strategies for a skill.

    Args:
        skill_name: Name of the skill
        k: Number of recent memories to retrieve

    Returns:
        Formatted memories as markdown
    """
    skill_path = find_skill_path(skill_name)
    memories_file = skill_path / 'memory' / 'memories.yaml'

    if not memories_file.exists():
        return f"No memories found for skill '{skill_name}'"

    with open(memories_file) as f:
        memories = yaml.safe_load(f) or []

    # Get k most recent
    recent = sorted(memories, key=lambda m: m.get('timestamp', ''), reverse=True)[:k]

    result = f"## Learned Strategies for {skill_name}\n\n"
    for m in recent:
        result += f"### {m['title']}\n{m['content']}\n\n"

    return result


@server.tool()
def add_skill_memory(
    skill_name: str,
    title: str,
    content: str,
    source: str = "self_execution",
    outcome: str = "success"
) -> str:
    """Record a learned strategy for a skill.

    Args:
        skill_name: Name of the skill
        title: Brief identifier for the memory
        content: The learned strategy or insight
        source: How it was learned (self_execution, human_feedback)
        outcome: What kind of learning (success, failure, correction)

    Returns:
        Confirmation message
    """
    import uuid
    from datetime import datetime

    skill_path = find_skill_path(skill_name)
    memories_file = skill_path / 'memory' / 'memories.yaml'
    memories_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    if memories_file.exists():
        with open(memories_file) as f:
            memories = yaml.safe_load(f) or []
    else:
        memories = []

    # Add new memory
    memories.append({
        'id': f"mem_{uuid.uuid4().hex[:8]}",
        'title': title,
        'content': content,
        'source': source,
        'outcome': outcome,
        'timestamp': datetime.now().isoformat()
    })

    with open(memories_file, 'w') as f:
        yaml.dump(memories, f)

    return f"Learned: {title}"


@server.tool()
def list_skill_memories(skill_name: str) -> str:
    """List all memories for a skill.

    Args:
        skill_name: Name of the skill

    Returns:
        List of memory titles with metadata
    """
    skill_path = find_skill_path(skill_name)
    memories_file = skill_path / 'memory' / 'memories.yaml'

    if not memories_file.exists():
        return f"No memories for skill '{skill_name}'"

    with open(memories_file) as f:
        memories = yaml.safe_load(f) or []

    result = f"## Memories for {skill_name} ({len(memories)} total)\n\n"
    for m in memories:
        result += f"- **{m['title']}** ({m['source']}, {m['outcome']})\n"

    return result
```

**Claude Code Configuration (.claude/settings.json):**
```json
{
  "mcpServers": {
    "skill-memory": {
      "command": "python",
      "args": ["/path/to/skill_memory_server.py"]
    }
  }
}
```

**SKILL.md with MCP tools:**
```markdown
---
name: code-reviewer
description: Reviews code with learning capabilities
---

# Code Reviewer Skill

## Before Starting

Use the `get_skill_memories` tool to retrieve learned strategies:
```
get_skill_memories("code-reviewer", k=3)
```

Apply any relevant strategies to your review.

## Instructions
...

## After Completion

If you learned something valuable, record it:
```
add_skill_memory(
    skill_name="code-reviewer",
    title="Brief title",
    content="What you learned",
    source="self_execution",
    outcome="success"
)
```
```

**Pros:**
- Structured tool interface
- Works across all skills uniformly
- Can add sophisticated retrieval later
- Memories managed externally from skill files

**Cons:**
- Requires MCP server setup
- More complex infrastructure
- Claude must remember to call tools

---

### Strategy 4: Claude Code Hooks (If Available)

If Claude Code supports hooks (pre/post skill activation):

**~/.claude/hooks/skill_activate.py:**
```python
"""Hook that runs when a skill is activated."""

def pre_activate(skill_name: str, skill_path: str) -> dict:
    """Augment skill context with memories."""
    from pathlib import Path
    import yaml

    memories_file = Path(skill_path) / 'memory' / 'memories.yaml'

    if not memories_file.exists():
        return {}

    with open(memories_file) as f:
        memories = yaml.safe_load(f) or []

    # Return context to inject
    if memories:
        recent = sorted(memories, key=lambda m: m.get('timestamp', ''), reverse=True)[:3]
        memory_text = "\n\n## Learned Strategies\n\n"
        for m in recent:
            memory_text += f"### {m['title']}\n{m['content']}\n\n"

        return {"inject_after_instructions": memory_text}

    return {}


def post_execute(skill_name: str, skill_path: str, result: dict) -> None:
    """Analyze execution for learning opportunities."""
    # Could analyze result for success/failure patterns
    # and prompt user or auto-extract memories
    pass
```

**Pros:**
- Automatic, seamless integration
- No changes to SKILL.md needed
- Can intercept and analyze all skill usage

**Cons:**
- Depends on Claude Code hook support (may not exist yet)
- Less transparent to user

---

## Recommended Approach for Claude Code

**Start with Strategy 1 (Self-Referential)** because:
1. Works today with no additional setup
2. Transparent—user can see exactly what Claude reads/writes
3. Human-editable memories
4. Natural fit for Claude's file-based workflow

**Evolve to Strategy 3 (MCP)** when:
1. You want structured memory management
2. Multiple skills share memory patterns
3. You need sophisticated retrieval
4. You want to track memory effectiveness

## Example: Complete EvolvingSkill for Claude Code

```
~/.claude/skills/code-reviewer/
├── SKILL.md
├── references/
│   └── review-checklist.md
└── memory/
    └── learned-strategies.md
```

**SKILL.md:**
```markdown
---
name: code-reviewer
description: Reviews code for bugs, style, and best practices. Learns and improves from each review.
allowed-tools: python
---

# Code Reviewer Skill

> An evolving code review assistant that learns from experience

## Setup

Before starting a review, check for learned strategies:

1. Read `memory/learned-strategies.md` if it exists
2. Keep these patterns in mind during your review
3. They represent insights from past reviews

## Review Process

### 1. Understand the Code
- What does this code do?
- What's the broader context?

### 2. Check Against Learned Patterns
Apply any relevant learned strategies from memory.

### 3. Analyze For Issues

**Bugs & Logic Errors:**
- Off-by-one errors
- Null/undefined handling
- Race conditions
- Edge cases

**Style & Readability:**
- Naming conventions
- Code organization
- Comment quality

**Security:**
- Input validation
- Injection vulnerabilities
- Authentication/authorization

**Performance:**
- Unnecessary computation
- Memory leaks
- N+1 queries

### 4. Provide Feedback
- Be specific (line numbers, examples)
- Explain *why* something is an issue
- Suggest concrete fixes
- Prioritize (critical > warning > nitpick)

## Learning

After completing a review, if you discovered something worth remembering:

### When to Record a Memory

- You found a bug pattern you've seen before
- You learned something about this codebase's conventions
- The user corrected your review (record the correction)
- You made a mistake (record how to avoid it)

### How to Record

Append to `memory/learned-strategies.md`:

```markdown
---

### [Brief Title]

**When**: [Context where this applies]

**Strategy**: [What to do]

**Why**: [Brief explanation]

*Learned: [date] from [brief context]*
```

### Example Memory Entry

```markdown
---

### Check for timezone handling in date comparisons

**When**: Reviewing code that compares dates or timestamps

**Strategy**: Verify that timezone is explicitly handled. Look for:
- datetime.now() vs datetime.utcnow()
- Naive vs aware datetime objects
- Database timezone settings

**Why**: Timezone bugs are subtle and often only appear in production
when servers are in different timezones than developers.

*Learned: 2025-12-29 from review where date comparison worked locally but failed in CI*
```

## References

- `references/review-checklist.md` - Detailed checklist for thorough reviews
```

**memory/learned-strategies.md (starts empty or with seeds):**
```markdown
# Learned Strategies

Patterns and insights learned from past code reviews.

---

### Verify error handling covers all exception types

**When**: Reviewing try/catch blocks

**Strategy**: Check that catch blocks handle specific exceptions, not just
generic Exception. Look for swallowed exceptions that hide bugs.

**Why**: Generic exception handling often masks the real problem and makes
debugging harder.

*Learned: 2025-12-01 from initial skill setup (seeded)*
```

## Learning Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SKILL ACTIVATED                                          │
│    Claude reads SKILL.md, sees instruction to check memory  │
├─────────────────────────────────────────────────────────────┤
│ 2. MEMORIES LOADED                                          │
│    Claude reads memory/learned-strategies.md                │
│    Patterns now in context for this session                 │
├─────────────────────────────────────────────────────────────┤
│ 3. SKILL EXECUTED                                           │
│    Claude performs review, applying learned patterns        │
├─────────────────────────────────────────────────────────────┤
│ 4. LEARNING CAPTURED                                        │
│    If Claude discovers new pattern or user corrects:        │
│    Claude appends to memory/learned-strategies.md           │
├─────────────────────────────────────────────────────────────┤
│ 5. NEXT ACTIVATION                                          │
│    New memories available for future reviews                │
└─────────────────────────────────────────────────────────────┘
```

## User Interaction Patterns

### Teaching the Skill

User can explicitly teach the skill:

```
User: "Remember that in this codebase, we always use arrow functions for callbacks"

Claude: I'll add that to the learned strategies.
[Appends to memory/learned-strategies.md]
```

### Correcting the Skill

When Claude's review misses something:

```
User: "You missed that this function can throw - always check for error handling"

Claude: Good catch! I'll remember to check for throw statements when reviewing.
[Appends correction to memory/learned-strategies.md]
```

### Reviewing Memories

User can ask to see what the skill has learned:

```
User: "What have you learned about reviewing code?"

Claude: [Reads memory/learned-strategies.md]
Here are the patterns I've learned...
```

### Pruning Memories

User can edit the file directly or ask Claude:

```
User: "That pattern about semicolons isn't relevant anymore, we use a formatter now"

Claude: I'll remove that from the learned strategies.
[Edits memory/learned-strategies.md]
```

## Limitations

1. **No Automatic Feedback Detection**: Claude must be told when to learn (unlike solveit where we can detect edits)

2. **Memory Quality**: Depends on Claude following instructions to record useful memories

3. **No Retrieval Filtering**: All memories loaded every time (fine for <50 entries)

4. **Single File**: Simple but doesn't scale to hundreds of memories

## Future Enhancements

1. **Structured YAML + Rendered Markdown**: Store in YAML, render to MD for reading

2. **MCP Memory Server**: Add when memory management becomes complex

3. **Cross-Skill Learning**: Share patterns between related skills

4. **Memory Effectiveness Tracking**: Record which memories actually helped

## Related Documents

- [EvolvingSkill Design](./evolvingskill-design.md) - Full architecture
- [Agent Skills Specification](https://agentskills.io/specification) - Base skill format
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code reference
