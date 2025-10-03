# CS450
# Instructor: Chiké Abuah
# Lab 1: Getting Started with Ollama & Prompt Engineering Basics
  
**Lab Duration:** ~45 minutes  

---

## Learning Objectives

By the end of this lab, you will be able to:
- Connect to the Ollama API using the Python client library
- Send basic requests to a language model
- Begin to understand model parameters (temperature, top-k, top-p)

---

## Prerequisites

- Python 
- Network Access to the course Ollama server (Via CS Lab)
- No prior context is required for this lab, although reading Lee Boonstra's *Prompt Engineering* would help with comprehension.

---


## Resources

- UV docs: https://docs.astral.sh/uv/
- Ollama Python library docs: https://github.com/ollama/ollama-python
- Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

---

## Part 0: Setup (10 minutes)

### Server Connection Information

- **Server URL**: `http://ollama.cs.wallawalla.edu:11434`
- **Available models**: cs450

### Install Required Libraries

I highly recommend `uv`. It (according to their docs): 

```markdown
🚀 Is a single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
⚡️ Is 10-100x faster than pip.
🗂️ Provides comprehensive project management, with a universal lockfile.
❇️ Runs scripts, with support for inline dependency metadata.
🐍 Installs and manages Python versions.
```

# Create your workspace and virtual environment

Open your Terminal.

Run the following command:

```bash
unset LD_PRELOAD
```

And don't ask me why! 😅

**Then**

```bash
mkdir -p cs450/lab1
cd cs450/lab1
uv init
uv add ollama
```

### Configure Server Connection

The Ollama Python library can connect to remote servers by setting the host URL.

Create a file called `test_connection.py`:

```python
import ollama

# Replace with your server URL
client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')

def test_connection():
    try:
        # List available models
        models = client.list()
        print("🎉 Connected successfully!")
        print("\nAvailable models:")
        for model in models['models']:
            print(f"  - {model['model']}")
        return True
    except Exception as e:
        print(f"✗ Error connecting: {e}")
        return False

if __name__ == "__main__":
    test_connection()
```

Run it:
```bash
uv run test_connection.py
```

**Expected output:**
```markdown
🎉 Connected successfully!

Available models:
  - t1c/deepseek-math-7b-rl:latest
  - solobsd/llemma-7b:latest
  - qwen2-math:latest
  - cs450:latest
  - cs396:latest
  - cs141:latest
  - qwen2.5-coder:latest
  - qwen3:latest
  - qwen2.5:latest
  - llama3.2:3b
```

---

## Part 1: Basic API Usage (10 minutes)

### 1.1 Create a Helper Module

Create `ollama_client.py`:

```python
import ollama

# Configure your server URL here
SERVER_HOST = 'http://ollama.cs.wallawalla.edu:11434'
client = ollama.Client(host=SERVER_HOST)

def call_ollama(prompt, model="cs450", **options):
    """
    Send a prompt to the Ollama API.
    
    Args:
        prompt (str): The prompt to send
        model (str): Model name to use
        **options: Additional model parameters (temperature, top_k, etc.)
    
    Returns:
        str: The model's response
    """
    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options=options
        )
        return response['response']
    
    except Exception as e:
        return f"Error: {e}"

def chat_ollama(messages, model="cs450", **options):
    """
    Send a chat conversation to the Ollama API.
    
    Args:
        messages (list): List of message dicts with 'role' and 'content'
        model (str): Model name to use
        **options: Additional model parameters
    
    Returns:
        str: The model's response
    """
    try:
        response = client.chat(
            model=model,
            messages=messages,
            options=options
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error: {e}"

def stream_ollama(prompt, model="cs450", **options):
    """
    Stream a response from Ollama (for real-time output).
    
    Args:
        prompt (str): The prompt to send
        model (str): Model name to use
        **options: Additional model parameters
    
    Yields:
        str: Chunks of the response as they arrive
    """
    try:
        stream = client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options=options
        )
        for chunk in stream:
            yield chunk['response']
    
    except Exception as e:
        yield f"Error: {e}"

# Test the function
if __name__ == "__main__":
    test_prompt = "Say 'Hello, World!' and nothing else."
    print("Testing API call...")
    result = call_ollama(test_prompt, temperature=0.1)
    print(f"Response: {result}")
    
    print("\n" + "="*50)
    print("Testing streaming:")
    for chunk in stream_ollama("Count to 5 slowly.", temperature=0.1):
        print(chunk, end='', flush=True)
    print()
```

### 1.2 Run Your First Prompt

Test your helper function:

```bash
uv run ollama_client.py
```

**Checkpoint:** You should see "Hello, World!" and then a count to 5.

---

## Part 2: Understanding Model Parameters (15 minutes)

### 2.1 Temperature Experiments

Create `temperature_test.py`:

```python
from ollama_client import call_ollama

prompt = "Complete this sentence: The weather today is"

print("Testing different temperatures:\n")

for temp in [0.0, 0.5, 1.0, 1.5]:
    print(f"Temperature: {temp}")
    response = call_ollama(
        prompt, 
        temperature=temp, 
        num_predict=20
    )
    print(f"Response: {response}\n")
```

**Task 2.1:** Run this script 3 times. What do you notice?

---

### 2.2 Top-K and Top-P

Create `sampling_test.py`:

```python
from ollama_client import call_ollama

prompt = "Write a creative opening line for a story:"

print("Testing sampling parameters:\n")

# Test 1: Low top-k (focused)
print("1. Low top-k (focused selection):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_k=10, 
    num_predict=30
)
print(f"{response}\n")

# Test 2: High top-k (diverse)
print("2. High top-k (diverse selection):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_k=50, 
    num_predict=30
)
print(f"{response}\n")

# Test 3: Low top-p (conservative)
print("3. Low top-p (conservative):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_p=0.5, 
    num_predict=30
)
print(f"{response}\n")

# Test 4: High top-p (exploratory)
print("4. High top-p (exploratory):")
response = call_ollama(
    prompt, 
    temperature=0.8, 
    top_p=0.95, 
    num_predict=30
)
print(f"{response}\n")
```

**Task 2.2:** Run this a few times and compare outputs.

---

## Part 3: Putting it all Together (10 minutes)

### 3.1 Final Experiments 


Create `story.py`:

```python
from ollama_client import call_ollama

prompt = "Tell me a cool story"

response = call_ollama(
    prompt, 
    temperature=0.1, 
    top_p=20,
    top_k=0.9,
    num_predict=40
)

print(f"Response: {response}\n")
```

**Run this program a few times and save the results to a file `bad_story.md`**

*then change `temperature=0.9,top_p=40,top_k=0.99`*,

**Run this new version of the program a few times and save the results to a file `good_story.md`**

*save both files to your `lab1` directory.

Create a file `lab1_results.md` in `lab1` and add:

```markdown
# Names: Your name and your lab partner's name
# Lab: Lab1 (Prompt Engineering Basics)
# Date: Today's date
```

Now, answer **(without using GenAI)**:

**Questions to answer:**
1. Which values for the prompt API parameters produced the most consistent results?
2. Which produced the most creative/varied results?
3. When would you use temp=0.0 vs temp=1.5? Describe a few specific situations.

*There's no right or wrong answers here, I just want to see some thought go into the response.*


### 3.2 Save your work and submit

Create a new **public** Github Repository called `cs450`, upload your local `cs450` folder there.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the first lab! 🎉