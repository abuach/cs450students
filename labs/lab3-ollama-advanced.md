# Lab 3: Advanced Prompting Techniques

**Course:** Advanced Prompt Engineering with Ollama 
**Lab Duration:** 65 minutes

---

## Learning Objectives

By the end of this lab, you will be able to:
- Use system prompts to control model behavior
- Apply Chain of Thought (CoT) reasoning to complex problems
- Understand flaws in LLM reasoning
- Chain multiple prompts together for complex tasks
- Manage context effectively in longer conversations
- Combine techniques for powerful prompt engineering

---

## Prerequisites

- Completed Lab 2
- Working `ollama_client.py` from Lab 2

---

## Part 0: Setup (5 minutes)

Run the following commands. 

If you were "driving" last time you only need to run:

- The `unset LD_PRELOAD` command
- The `cd cs450` and then `uv add pygame` commands.

```bash
unset LD_PRELOAD
cd cs450
uv init 
uv add ollama 
uv add pygame
mkdir util 
cp lab1/ollama_client.py util/
```

Create a file `__init__.py` in `cs450/util` and add:

```python
from .ollama_client import call_ollama, stream_ollama, chat_ollama

__all__ = ['call_ollama', 'stream_ollama', 'chat_ollama']

package_version = "1.0.0"

print(f"Initializing UTIL version {package_version}")
```

*Then run:*

```bash
mkdir lab3
cd lab3
```
---

## Part 1: System Prompting (5 minutes)

### 1.1 Understanding System Prompts

System prompts set the **overall behavior** and constraints for the model.

Create `system_prompting.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import chat_ollama

def test_system_prompt(user_query, system_message):
    """Test different system prompts."""
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_query}
    ]
    return chat_ollama(messages, temperature=0.3)

# Test 1: No system prompt (baseline)
query = "Explain quantum entanglement."
print("No System Prompt:")
print(test_system_prompt(query, ""))
print("\n" + "="*60 + "\n")

# Test 2: Concise system prompt
print("System: Be concise (max 2 sentences)")
result = test_system_prompt(
    query, 
    "You provide concise explanations. Maximum 2 sentences."
)
print(result)
print("\n" + "="*60 + "\n")

# Test 3: ELI5 system prompt
print("System: Explain like I'm 5")
result = test_system_prompt(
    query,
    "You explain complex topics to 5-year-olds using simple words and analogies."
)
print(result)
```

**Reflection 1.1:** Run and compare outputs. Note how the system prompt changes responses.

---

## Part 2: Role Prompting (10 minutes)

### 2.1 Specialized Roles

Create `role_prompting.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import chat_ollama

def answer_as_role(question, role, expertise):
    """Answer questions from a specific role perspective."""
    system = f"You are a {role} with expertise in {expertise}."
    
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': question}
    ]
    
    return chat_ollama(messages, temperature=2.0)

question = "In one paragraph, how can I use software engineering to become very respected and/or wealthy?"

# Test different roles
roles = [
    ("university professor", "formal computer science education & academia"),
    ("senior software engineer", "cryptocurrency"),
    ("tech hobbyist", "GenAI")
]

for role, expertise in roles:
    print(f"\nRole: {role}")
    print(f"Expertise: {expertise}")
    print("-" * 60)
    answer = answer_as_role(question, role, expertise)
    print(answer)
    print("\n" + "="*60)
```

**Reflection 2.1:** Can you think of other roles that could provide an interesting perspective?

---

## Part 3: Chain of Thought Reasoning (15 minutes)

### 3.2 Few-Shot Chain of Thought

Create `chain_of_thought2.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def solve_with_fewshot_cot(problem):
    """Use few-shot examples to guide reasoning."""
    prompt = f"""Solve word problems step by step.

Example:
Problem: A train travels 60 mph for 2 hours. How far does it go?
Solution:
Step 1: Speed = 60 mph
Step 2: Time = 2 hours
Step 3: Distance = Speed × Time = 60 × 2 = 120 miles
Answer: 120 miles

Example:
Problem: Jane has $50. She spends $15 on lunch and $20 on a book. How much remains?
Solution:
Step 1: Starting amount = $50
Step 2: Total spent = $15 + $20 = $35
Step 3: Remaining = $50 - $35 = $15
Answer: $15

Now solve just like my examples:
Problem: {problem}
Solution:"""
    
    return call_ollama(prompt, temperature=0.0)

problems = [
    "A rectangle is 8 cm long and 5 cm wide. What is its perimeter?",
    "Bob earns $25/hour and works 6 hours. He pays $30 in taxes. What's his take-home?"
]

print("\nFew-Shot CoT:")
for i, prob in enumerate(problems, 1):
    print(f"\nProblem {i}:")
    print(solve_with_fewshot_cot(prob))
    print("="*60)
```

---

### 3.2 Misguided Attention

Create `misguided.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def solve_with_cot(problem):
    """Solve using Chain of Thought."""
    prompt = f"""{problem}

Let's solve this step by step:"""
    
    return call_ollama(prompt, temperature=0.0)


# Logic problem
problem = """A farmer is on one side of a river with a wolf, a goat, and a cabbage. When he is crossing the river in a boat, he can only take one item with him at a time. The wolf will eat the goat if left alone together, and the goat will eat the cabbage if left alone together. How can the farmer transport the goat across the river without it being eaten?"""

print("\n" + "="*60)
print("\nProblem (Logic):")
print(solve_with_cot(problem))
```

**Task 3.2:** Read the problem statement here: https://en.wikipedia.org/wiki/Wolf,_goat_and_cabbage_problem 

Look at the problem statement in the prompt above. 

Do you notice any difference? Did the LLM provide an *efficient* response?

---

## Part 4: Prompt Chaining (10 minutes)

### 4.1 Sequential Processing

Create `prompt_chaining.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def analyze_text_chain(text):
    """Chain prompts to analyze text progressively."""
    
    # Step 1: Summarize
    summary_prompt = f"Summarize this in one sentence:\n\n{text}"
    summary = call_ollama(summary_prompt, temperature=0.3, num_predict=100)
    print("Step 1 - Summary:")
    print(summary)
    
    # Step 2: Extract key points from summary
    keypoints_prompt = f"List 3 key points from: {summary}"
    keypoints = call_ollama(keypoints_prompt, temperature=0.3)
    print("\nStep 2 - Key Points:")
    print(keypoints)
    
    # Step 3: Generate questions
    questions_prompt = f"Generate 2 questions someone might ask about:\n{keypoints}"
    questions = call_ollama(questions_prompt, temperature=0.5)
    print("\nStep 3 - Questions:")
    print(questions)
    
    return {
        'summary': summary,
        'keypoints': keypoints,
        'questions': questions
    }

# Test text
article = """
Artificial intelligence is transforming healthcare through advanced 
diagnostic tools. Machine learning algorithms can now detect diseases 
from medical images with accuracy matching expert radiologists. This 
technology reduces diagnosis time and helps doctors make better treatment 
decisions. However, challenges remain in data privacy and algorithm bias.
"""

print("Chained Analysis:\n" + "="*60 + "\n")
result = analyze_text_chain(article)
```

---

### 4.2 Refinement Chain

Create `prompt_chaining2.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def refine_writing_chain(draft, style="professional"):
    """Progressively refine writing through multiple steps."""
    
    # Step 1: Fix grammar
    grammar_prompt = f"Fix any grammar errors:\n\n{draft}"
    fixed = call_ollama(grammar_prompt, temperature=0.2)
    print("Step 1 - Grammar Fixed:")
    print(fixed)
    
    # Step 2: Improve clarity
    clarity_prompt = f"Make this clearer and more concise:\n\n{fixed}"
    clear = call_ollama(clarity_prompt, temperature=0.3)
    print("\nStep 2 - Clarity Improved:")
    print(clear)
    
    # Step 3: Apply style
    style_prompt = f"Rewrite in {style} style:\n\n{clear}"
    final = call_ollama(style_prompt, temperature=0.4)
    print(f"\nStep 3 - {style.title()} Style:")
    print(final)
    
    return final

draft = """me and my team was working on the project 
but we didnt finish it because of there was problems"""

print("\n" + "="*60)
print("\nRefinement Chain:\n" + "="*60 + "\n")
refine_writing_chain(draft, "professional")
```

---

## Part 5: Context Management (10 minutes)

### 5.1 Managing Conversation History

Create `context.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import chat_ollama

class ConversationManager:
    """Manage conversation context efficiently."""
    
    def __init__(self, system_prompt="", max_history=10):
        self.system_prompt = system_prompt
        self.messages = []
        self.max_history = max_history
        
        if system_prompt:
            self.messages.append({
                'role': 'system',
                'content': system_prompt
            })
    
    def add_user_message(self, content):
        """Add user message to history."""
        self.messages.append({
            'role': 'user',
            'content': content
        })
        self._trim_history()
    
    def get_response(self, temperature=0.7):
        """Get model response and add to history."""
        response = chat_ollama(self.messages, temperature=temperature)
        self.messages.append({
            'role': 'assistant',
            'content': response
        })
        self._trim_history()
        return response
    
    def _trim_history(self):
        """Keep only recent messages."""
        # Keep system prompt + last N messages
        if len(self.messages) > self.max_history + 1:
            system = self.messages[0] if self.messages[0]['role'] == 'system' else None
            recent = self.messages[-(self.max_history):]
            self.messages = ([system] if system else []) + recent
    
    def get_summary(self):
        """Summarize conversation so far."""
        convo = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in self.messages if m['role'] != 'system'
        ])
        return convo

# Test
conversation = ConversationManager(
    system_prompt="You are a helpful Python tutor.",
    max_history=6
)

conversation.add_user_message("What is a dictionary?")
print("User: What is a dictionary?")
response = conversation.get_response()
print(f"Assistant: {response}\n")

conversation.add_user_message("How do I add items to it?")
print("User: How do I add items to it?")
response = conversation.get_response()
print(f"Assistant: {response}\n")

conversation.add_user_message("Show me an example")
print("User: Show me an example")
response = conversation.get_response()
print(f"Assistant: {response}")
```

---

### 5.2 Context Summarization

Create `context2.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def compress_context(long_context):
    """Compress long context to key information."""
    prompt = f"""Extract only the essential information from this text.
Keep names, numbers, and key facts. Remove filler.

Text: {long_context}

Essential information:"""
    
    return call_ollama(prompt, temperature=0.2, num_predict=150)

long_text = """
Yesterday, I went to the grocery store, which was quite crowded 
actually, and I ran into my old friend Sarah Johnson. We chatted for 
a while about various things. She mentioned she got a new job at 
TechCorp starting next Monday with a salary of $85,000. She seemed 
really excited about it. The position is Senior Data Analyst. We 
agreed to meet for coffee next Friday at 3pm at Starbucks downtown.
"""

print("\n" + "="*60)
print("\nContext Compression:")
print("\nOriginal:", long_text)
print("\nCompressed:", compress_context(long_text))
```

---

## Part 6: Step-back Prompting (10 minutes)

### 6.1 Putting it all together 

For this part you can use https://ai.cs.wallawalla.edu/ if it's more convenient.

If you chose to use the Python API, this will allow you to change the parameters like temperature.

If you choose to use the Python API, you can consider using the `ConversationManager` from `Part 5.1`.

Create `snake1.py`:

Ask the LLM to: `"Create a Python snake game in Pygame in under 400 lines of code"`

**Task 6.1.1:** Run the program and try playing the game. Does it work correctly?

Now ask: `"Based on popular implementations of the classic game Snake, what are 2 of the best customizations for this game? Do not provide any examples."`

Note the suggestions, then ask: `"Create a Python snake game which implements those features in Pygame in under 500 lines of code"`

**Task 6.1.2:** Create `snake2.py`. Run the new Snake game. How is it different from the first version? 

---

## Optional: Top 500

Using all the prompt engineering techniques you have learned so far, create the best possible version of `Snake` in `Pygame` in under `500` lines of code.

## Submission Requirements

Create a file `lab3_results.md` in `lab3` and add:

```markdown
# Names: Your name and your lab partner's name
# Lab: Lab3 (Advanced Prompt Engineering)
# Date: Today's date

Submit:
1. Your full `lab3` directory including all the code you ran and,
2. Your `lab3_results.md` with:
   - Your **Task 3.2** response
   - Your **Task 6.1.2:** response

Push all the code and md files to your remote Github `cs450` repo if you already have one.

OR if you don't:

Create a new **public** Github Repository called `cs450`, upload your local `cs450` folder there.

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Either way, email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

Email subject should be `CS450 LAB 3`

Great, you're done with the third lab!

---


