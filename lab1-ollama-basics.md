# Lab 1: Getting Started with Ollama & Prompt Engineering
  
**Lab Duration:** 45 minutes  

---

## Learning Objectives

By the end of this lab, you will be able to:
- Connect to the Ollama API using the Python client library
- Send basic requests to a language model
- Understand model parameters (temperature, top-k, top-p)

---

## Prerequisites

- Python 
- Network Access to the course Ollama server (Via CS Lab)

---


## Resources

- Reading on prompt engineering
- Ollama Python library docs: https://github.com/ollama/ollama-python
- Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

---

## Part 0: Setup (10 minutes)

### Server Connection Information

- **Server URL**: `http://ollama.cs.wallawalla.edu:11434`
- **Available models**: cs450

### Install Required Libraries

I highly recommend `uv`

```bash
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
        print("✓ Connected successfully!")
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
```
✓ Connected successfully!

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

## Part 1: Basic API Usage (15 minutes)

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

**✓ Checkpoint:** You should see "Hello, World!" and then a count to 5.

---

## Part 2: Understanding Model Parameters (20 minutes)

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

## Document Your Findings (10 minutes)

Create a new directory `cs450`

Create a file `lab1_results.md` in `cs450` and add:

```

# Names: Your name and your lab partner's name
# Lab: Lab1 (Prompt Engineering Basics)
# Date: Today's date


```

Now, answer:

**Questions to answer:**
1. Which temperature gives the most consistent results?
2. Which gives the most creative/varied results?
3. When would you use temp=0.0 vs temp=1.5?
4. What do you think the difference is between low top-k and high top-k?
5. What do you think the difference is between low top-p and high top-p?


Create a new public Github Repository called `cs450`, upload your local `cs450` there.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`