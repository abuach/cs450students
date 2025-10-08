# Lab 2: Intermediate Ollama & Prompt Engineering
 
**Lab Duration:** 50 minutes  

---

## Learning Objectives

By the end of this lab, you will be able to:
- Apply zero-shot and few-shot prompting techniques
- Experiment with different prompt styles
- Evaluate and iterate on prompt quality

---

## Prerequisites

- Python 
- Network Access to the course Ollama server (Via CS Lab)
- Completed Lab 1

---


## Resources

- Ollama Python library docs: https://github.com/ollama/ollama-python
- Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
- https://docs.ollama.com/ 
- https://docs.ollama.com/api#generate-a-chat-completion 

---

## Part 1: Setup (5 minutes)

Run the following commands: 

```bash
unset LD_PRELOAD
cd cs450
uv init 
uv add ollama 
mkdir util 
```

If you "drove" for Lab 1 you can run:

```bash
cp lab1/ollama_client.py util/
```

Otherwise manually copy the client from the `lab1` specification.


Create a file `__init__.py` in `cs450/util` and add:

```python
from .ollama_client import call_ollama, stream_ollama, chat_ollama

__all__ = ['call_ollama', 'stream_ollama', 'chat_ollama']

package_version = "1.0.0"

print(f"Initializing UTIL version {package_version}")
```

*Then run:*

```bash
mkdir lab2
cd lab2
```

Remember to use `uv run [program].py` to run your code.
---


## Part 2: Zero-Shot Prompting (10 minutes)

### 2.1 Simple Classification

Create `zero_shot_class.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def classify_sentiment(review):
    """Classify movie review sentiment using zero-shot prompting."""
    prompt = f"""Classify this movie review as POSITIVE, NEGATIVE, or NEUTRAL.
Return only the classification label.

Review: {review}

Classification:"""
    
    response = call_ollama(
        prompt, 
        temperature=0.1, 
        num_predict=5
    )
    return response.strip()

# Test cases
reviews = [
    "This movie was absolutely amazing! Best film of the year!",
    "Terrible waste of time. I want my money back.",
    "It was okay. Nothing special but not bad either.",
    "A masterpiece of cinema. Brilliantly directed and acted.",
    "Boring and predictable. Fell asleep halfway through."
]

print("Zero-Shot Sentiment Classification\n" + "="*50)
for review in reviews:
    sentiment = classify_sentiment(review)
    print(f"\nReview: {review[:50]}...")
    print(f"Sentiment: {sentiment}")
```

**Task 2.1:** Run this and evaluate accuracy.

**Reflections:**
1. Did it classify all reviews correctly?

---

### 2.2 Information Extraction

Create `zero_shot_extract.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def extract_info(text):
    """Extract structured information using zero-shot prompting."""
    prompt = f"""Extract the person's name, age, and occupation from this text.
Return as JSON.

Text: {text}

JSON:"""
    
    response = call_ollama(
        prompt, 
        temperature=0.1, 
        num_predict=100
    )
    return response

# Test
text = "My name is Alice Johnson, I'm 28 years old, and I work as a software engineer."
result = extract_info(text)
print("\n\nInformation Extraction\n" + "="*50)
print(f"Input: {text}")
print(f"Output: {result}")
```

**Task 2.2:** Test with different texts:
- "Bob Smith, age 35, teacher"
- "Dr. Carol Williams is a 42-year-old physician"
- "I'm Dave, 29, and I do marketing"

---

## Part 3: Few-Shot Prompting (10 minutes)

### 3.1 Email Classification

Create `few_shot_class.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def classify_email_fewshot(email_body):
    """Classify emails as SPAM, IMPORTANT, or NORMAL using few-shot."""
    
    prompt = f"""Classify emails as SPAM, IMPORTANT, or NORMAL.

Example 1:
Email: "Congratulations! You've won $1,000,000! Click here now!"
Classification: SPAM

Example 2:
Email: "Meeting with CEO rescheduled to tomorrow 9am. Please confirm."
Classification: IMPORTANT

Example 3:
Email: "Weekly newsletter: Here are this week's top articles."
Classification: NORMAL

Now classify this email:
Email: {email_body}
Classification:"""
    
    response = call_ollama(
        prompt, 
        temperature=0.1, 
        num_predict=10
    )
    return response.strip()

# Test cases
emails = [
    "URGENT: Your account will be closed unless you verify now!",
    "Board meeting agenda attached. Review before Friday's meeting.",
    "Thanks for signing up for our service. Welcome!",
    "You are the lucky winner! Claim your prize within 24 hours!",
    "Quarterly results are in. Please review the attached report ASAP."
]

print("Few-Shot Email Classification\n" + "="*50)
for email in emails:
    classification = classify_email_fewshot(email)
    print(f"\nEmail: {email[:60]}...")
    print(f"Classification: {classification}")
```

**Task 3.11:** Run and evaluate.

**Reflections:**
1. How does this compare to zero-shot?
2. Do the guiding examples seem helpful?
3. What do you think would happen with different examples?

**Task 3.12:** Try changing some of the classification labels to be intentionally wrong.
    Rerun and evaluate. Does anything change?

---

### 3.2 Code Generation with Examples

Create `few_shot_codegen.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def generate_function_fewshot(description):
    """Generate Python functions using few-shot prompting."""
    
    prompt = f"""Generate Python functions based on descriptions.

Example 1:
Description: A function that checks if a number is even
Code:
def is_even(n):
    return n % 2 == 0

Example 2:
Description: A function that reverses a string
Code:
def reverse_string(s):
    return s[::-1]

Now generate:
Description: {description}
Code:"""
    
    response = call_ollama(
        prompt, 
        model="cs450", 
        temperature=0.2, 
        num_predict=150
    )
    return response

# Test
descriptions = [
    "A function that finds the maximum value in a list",
    "A function that counts vowels in a string",
    "A function that calculates factorial"
]

print("\n\nFew-Shot Code Generation\n" + "="*50)
for desc in descriptions:
    code = generate_function_fewshot(desc)
    print(f"\nDescription: {desc}")
    print(f"Generated Code:\n{code}\n")
```

**Task 3.21:** Run a couple times and look at the output.

**Task 3.22:** Now try removing the code examples from the prompt, turning it to zero-shot. 
   Rerun the code. Do you notice any difference in the generated code?

---

## Part 4: Prompt Engineering Practice (10 minutes)

### Exercise 4.1: Question Answering System

**Goal:** Build a simple Question Answering system using context.

**Requirements:**
- Provide context text
- Ask questions about the context
- Model should answer based ONLY on the context
- Return "I don't know" if answer isn't in context

**Starter code:**

Create `question.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import call_ollama

def answer_question(context, question):
    """
    Answer questions based on provided context.
    
    Args:
        context: Background information
        question: Question to answer
    """
    
    # TODO: Build your prompt
    # Make it clear to only use the context
    # Handle cases where answer isn't available
    
    prompt = f"""Your prompt here...
    
Context: {context}

Question: {question}

Answer:"""
    
    response = call_ollama(
        prompt, 
        temperature=0.1, 
        num_predict=100
    )
    return response

# Test context
context = """
Python is a high-level programming language created by Guido van Rossum 
in 1991. It emphasizes code readability and uses significant indentation. 
Python supports multiple programming paradigms including procedural, 
object-oriented, and functional programming. It is widely used in web 
development, data science, and automation.
"""

# Test questions
questions = [
    "Who created Python?",
    "What year was Python created?",
    "What is Python used for?",
    "What is Python's execution speed?"  # Not in context!
]

if __name__ == "__main__":
    for q in questions:
        answer = answer_question(context, q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
```

---

## Part 5: Working with Chat Format (10 minutes)

### 5.1 Multi-Turn Conversations

Create `chat.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import chat_ollama

def run_conversation():
    """Demonstrate multi-turn conversation."""
    
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful Python programming tutor.'
        },
        {
            'role': 'user',
            'content': 'What is a list comprehension?'
        }
    ]
    
    # First response
    response = chat_ollama(messages, temperature=0.3)
    print("Assistant:", response)
    
    # Add to conversation
    messages.append({
        'role': 'assistant',
        'content': response
    })
    
    messages.append({
        'role': 'user',
        'content': 'Can you show me an example?'
    })
    
    # Second response
    response = chat_ollama(messages, temperature=0.3)
    print("\nAssistant:", response)

if __name__ == "__main__":
    run_conversation()
```

**Task:** Modify this to create a 4-turn conversation about a topic of your choice.

---

## Part 6: Evaluation (10 minutes)

### 6.1 Document Your Findings

Create a file `lab2_results.md` in `lab2` and add:

```markdown
# Names: Your name and your lab partner's name
# Lab: Lab2 (Intermediate Prompt Engineering)
# Date: Today's date
```

Now, answer **(without using GenAI)**:

**1. Zero-Shot and Few-Shot:**
- When would you choose zero-shot?
- When is few-shot worth the extra effort?
- How did the classification output change from `Task 3.11` `Task 3.12`
- How did the generated code change from `Task 3.21` to `Task 3.22`?

---

## Submission

Stage and commit the following files to Git:
 
1. **lab2/** folder with:
   - **lab2_results.md** with your findings
   - `question.py` (Question Answering system)
   - `chat.py` (Interactive Conversation)

Push all the code and md files to your remote Github `cs450` repo if you already have one.

OR if you don't:

Create a new **public** Github Repository called `cs450`, upload your local `cs450` folder there.

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Either way, email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

Email subject should be `CS450 LAB 2`

Great, you're done with the second lab! 💪

Let me know when you're done.

---
