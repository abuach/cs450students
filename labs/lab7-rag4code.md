# Lab 7: Retrieval Augmented Generation for Code

**Lab Duration:** 80 minutes  

---

## Learning Objectives

By the end of this lab, you will understand how to:
- Use modern code-specific embedding models (UniXcoder) for code understanding
- Build a vector store with Chroma for efficient code retrieval
- Implement semantic search optimized for code snippets
- Parse code using AST to extract meaningful chunks
- Create a RAG system that retrieves relevant code context
- Generate code solutions using retrieved examples
- Understand how code-specific embeddings differ from generic text embeddings

---

## Prerequisites

- Python 3.10+
- Network Access to the course Ollama server (Via CS Lab)

---

## Resources

- UniXcoder: https://huggingface.co/microsoft/unixcoder-base
- Chroma documentation: https://docs.trychroma.com/
- Transformers library: https://huggingface.co/docs/transformers/
- Python AST documentation: https://docs.python.org/3/library/ast.html

---

## Why UniXcoder?

UniXcoder (2022) significantly outperforms older models:
- **Trained on 5 languages**: Python, Java, JavaScript, PHP, Ruby
- **Better semantic understanding**: Understands code semantics, not just syntax
- **Context-aware**: Considers surrounding code context
- **Modern architecture**: Benefits from recent advances in transformers

It's currently one of the best open-source models for code understanding tasks.

---


## Part 1: Setup (10 minutes)

Run the following commands: 

```bash
unset LD_PRELOAD
cd cs450
uv add chromadb
uv add transformers
uv add torch
mkdir lab7
cd lab7
```

**Note:** UniXcoder is a modern (2022) code model that significantly outperforms older models like CodeBERT for code understanding tasks.

Remember to use `uv run [program].py` to run your code.

---

## Part 2: Understanding Code Embeddings (25 minutes)

### 2.1 Load and Test UniXcoder

Create `unixcoder_intro.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModel

class UniXcoderEmbedder:
    """Wrapper for UniXcoder embeddings - a modern code understanding model."""
    
    def __init__(self):
        print("Loading UniXcoder model (this may take a minute on first run)...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        self.model.eval()
        print("UniXcoder loaded successfully!")
        print("UniXcoder is a 2022 model trained on code from 5+ languages.")
    
    def encode(self, text):
        """Generate embedding for text using mean pooling."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512,
                               padding=True)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling over all tokens (best practice for embeddings)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Weighted average using attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze().numpy()
        
        return embedding

# Test the embedder
if __name__ == "__main__":
    embedder = UniXcoderEmbedder()
    
    code = "def validate_email(email):\n    return '@' in email and '.' in email"
    
    print("\n" + "="*60)
    print("Testing UniXcoder Embeddings")
    print("="*60)
    print(f"\nCode: {code}")
    
    embedding = embedder.encode(code)
    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
```

**Task 2.1:** Run this to verify UniXcoder works correctly.

---

### 2.2 Compare Generic vs Code-Specific Embeddings

Create `embedding_comparison.py`:

```python
from sklearn.metrics.pairwise import cosine_similarity
import ollama

from unixcoder_intro import UniXcoderEmbedder

# Initialize models
print("Loading models...")
unixcoder = UniXcoderEmbedder()
ollama_client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')

def get_ollama_embedding(text):
    """Get generic text embedding from Ollama."""
    response = ollama_client.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return response['embedding']

# Test with clearly different code snippets
code_snippets = {
    'email_val_1': """def validate_email(email):
    '''Check if email is valid.'''
    return '@' in email and '.' in email.split('@')[1]""",
    
    'email_val_2': """def check_email(address):
    '''Verify email format.'''
    if '@' not in address:
        return False
    return '.' in address.split('@')[1]""",
    
    'math_calc': """def calculate_compound_interest(principal, rate, years):
    '''Calculate compound interest.'''
    return principal * ((1 + rate) ** years)""",
    
    'string_op': """def reverse_string(text):
    '''Reverse a string.'''
    return text[::-1]"""
}

print("\n" + "="*60)
print("Comparing Generic vs Code-Specific Embeddings")
print("="*60)

# Generate embeddings
print("\nGenerating UniXcoder embeddings...")
unix_embeddings = {}
for name, code in code_snippets.items():
    unix_embeddings[name] = unixcoder.encode(code).reshape(1, -1)

print("Generating generic embeddings...")
generic_embeddings = {}
for name, code in code_snippets.items():
    generic_embeddings[name] = [get_ollama_embedding(code)]

# Compare similar code (both email validation)
print("\n" + "="*60)
print("Test 1: Similar Functions (both email validation)")
print("="*60)
print("email_val_1 vs email_val_2")

unix_sim = cosine_similarity(
    unix_embeddings['email_val_1'], 
    unix_embeddings['email_val_2']
)[0][0]

generic_sim = cosine_similarity(
    generic_embeddings['email_val_1'], 
    generic_embeddings['email_val_2']
)[0][0]

print(f"UniXcoder:      {unix_sim:.4f}")
print(f"Generic model:  {generic_sim:.4f}")
print(f"Difference:     {unix_sim - generic_sim:+.4f}")

# Compare different code (email vs math)
print("\n" + "="*60)
print("Test 2: Different Functions (email vs math)")
print("="*60)
print("email_val_1 vs math_calc")

unix_diff = cosine_similarity(
    unix_embeddings['email_val_1'], 
    unix_embeddings['math_calc']
)[0][0]

generic_diff = cosine_similarity(
    generic_embeddings['email_val_1'], 
    generic_embeddings['math_calc']
)[0][0]

print(f"UniXcoder:      {unix_diff:.4f}")
print(f"Generic model:  {generic_diff:.4f}")
print(f"Difference:     {unix_diff - generic_diff:+.4f}")

# Compare very different code (email vs string)
print("\n" + "="*60)
print("Test 3: Very Different Functions (email vs string reversal)")
print("="*60)
print("email_val_1 vs string_op")

unix_very_diff = cosine_similarity(
    unix_embeddings['email_val_1'], 
    unix_embeddings['string_op']
)[0][0]

generic_very_diff = cosine_similarity(
    generic_embeddings['email_val_1'], 
    generic_embeddings['string_op']
)[0][0]

print(f"UniXcoder:      {unix_very_diff:.4f}")
print(f"Generic model:  {generic_very_diff:.4f}")
print(f"Difference:     {unix_very_diff - generic_very_diff:+.4f}")

# Summary
print("\n" + "="*60)
print("Analysis")
print("="*60)
print("A good code embedding model should:")
print("  - Give HIGH similarity for functionally similar code")
print("  - Give LOW similarity for functionally different code")
print(f"\nUniXcoder discrimination (similar - different): {unix_sim - unix_diff:.4f}")
print(f"Generic discrimination (similar - different):   {generic_sim - generic_diff:.4f}")
```

**Task 2.2:** Run and observe the differences.

**Reflections:**
1. Look at the "discrimination" scores - what do they tell you?

---

### 2.3 Test Code Structure Understanding

Create `structure_test.py`:

```python
from sklearn.metrics.pairwise import cosine_similarity

from unixcoder_intro import UniXcoderEmbedder

embedder = UniXcoderEmbedder()

# Test how UniXcoder handles code variations
test_cases = {
    "original": """def add_numbers(a, b):
    '''Add two numbers together.'''
    return a + b""",
    
    "renamed_vars": """def add_numbers(x, y):
    '''Add two numbers together.'''
    return x + y""",
    
    "different_name": """def sum_values(a, b):
    '''Add two numbers together.'''
    return a + b""",
    
    "with_typing": """def add_numbers(a: int, b: int) -> int:
    '''Add two numbers together.'''
    return a + b""",
    
    "different_logic": """def multiply_numbers(a, b):
    '''Multiply two numbers together.'''
    return a * b""",
    
    "completely_different": """def send_email(recipient, message):
    '''Send an email to recipient.'''
    print(f"Sending to {recipient}: {message}")"""
}

print("\n" + "="*60)
print("Testing UniXcoder's Code Understanding")
print("="*60)

# Get embeddings
embeddings = {name: embedder.encode(code).reshape(1, -1) 
              for name, code in test_cases.items()}

# Compare all against original
base = embeddings["original"]
print(f"\nBase code:\n{test_cases['original']}\n")
print("="*60)
print("Similarity Scores:")
print("="*60)

results = []
for name, code in test_cases.items():
    if name != "original":
        sim = cosine_similarity(base, embeddings[name])[0][0]
        results.append((name, sim))

# Sort by similarity (highest first)
results.sort(key=lambda x: x[1], reverse=True)

for name, sim in results:
    print(f"{sim:.4f} - {name}")

print("\n" + "="*60)
print("Observations:")
print("="*60)
print("Higher scores = more similar to original")
```

**Task 2.3:** Run and analyze how UniXcoder handles variations.

**Reflections:**
1. Does UniXcoder recognize functionally similar code despite variable name changes and type hints?


---

### 2.4 Operator Sensitivity Test

Create `operator_sensitivity.py`:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from unixcoder_intro import UniXcoderEmbedder

print("Loading UniXcoder...")
embedder = UniXcoderEmbedder()

# Original function
original = """def validate_age(age):
    '''Check if age meets minimum requirement.'''
    if age < 18:
        return False
    return True"""

# Test cases: progressively change operators
test_cases = {
    "original": {
        "code": """def validate_age(age):
    '''Check if age meets minimum requirement.'''
    if age < 18:
        return False
    return True""",
        "changes": "None (baseline)",
        "description": "Original: age < 18"
    },
    
    "one_operator": {
        "code": """def validate_age(age):
    '''Check if age meets minimum requirement.'''
    if age > 18:
        return False
    return True""",
        "changes": "1 operator: < → >",
        "description": "Changed: age > 18 (completely different logic)"
    },
    
    "two_operators": {
        "code": """def validate_age(age):
    '''Check if age meets minimum requirement.'''
    if age > 18:
        return True
    return False""",
        "changes": "2 operators: < → >, flip returns",
        "description": "Changed: age > 18, swapped True/False"
    },
    
    "three_operators": {
        "code": """def validate_age(age):
    '''Check if age meets minimum requirement.'''
    if age >= 21:
        return True
    return False""",
        "changes": "3 operators: < → >=, 18 → 21, flip returns",
        "description": "Changed: age >= 21, swapped True/False"
    },
    
    "cosmetic_only": {
        "code": """def validate_age(user_age):
    '''Check if age meets minimum requirement.'''
    if user_age < 18:
        return False
    return True""",
        "changes": "0 operators (cosmetic: age → user_age)",
        "description": "Only variable name changed"
    },
    
    "comment_only": {
        "code": """def validate_age(age):
    '''Check if age meets minimum requirement.'''
    # Checking minimum age limit
    if age < 18:
        return False
    return True""",
        "changes": "0 operators (added comment)",
        "description": "Only added comment"
    }
}

print("\n" + "="*70)
print("UniXcoder Operator Sensitivity Test")
print("="*70)
print("\nOriginal Function:")
print(original)
print("\n" + "="*70)

# Generate embeddings
embeddings = {}
for name, test in test_cases.items():
    embeddings[name] = embedder.encode(test['code']).reshape(1, -1)

# Calculate similarities to original
base = embeddings["original"]
results = []

for name, test in test_cases.items():
    if name != "original":
        sim = cosine_similarity(base, embeddings[name])[0][0]
        results.append({
            'name': name,
            'similarity': sim,
            'changes': test['changes'],
            'description': test['description']
        })

# Sort by similarity (highest first)
results.sort(key=lambda x: x['similarity'], reverse=True)

print("\nSimilarity to Original (sorted high to low):")
print("="*70)
print(f"{'Test Case':<20} {'Similarity':<12} {'Changes':<30}")
print("-"*70)

for r in results:
    print(f"{r['name']:<20} {r['similarity']:<12.4f} {r['changes']:<30}")

print("\n" + "="*70)
print("Detailed Analysis")
print("="*70)

for r in results:
    print(f"\n{r['name']}:")
    print(f"  Similarity: {r['similarity']:.4f}")
    print(f"  Changes: {r['changes']}")
    print(f"  Description: {r['description']}")


```

**Task 2.4:** Run and analyze how UniXcoder handles variations.

**Reflections:**
1. Does UniXcoder recognize increasingly numerous changes?
2. How does it handle the "cosmetic" change? Does it think it's important, and if so, why?

---

## 2.5 Comparison Challenge

Try running the same test with a generic text embedding model (like the Ollama embeddings from Lab 4) and compare:

```python
# Add to operator_sensitivity.py
import ollama

ollama_client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')

def get_ollama_embedding(text):
    response = ollama_client.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return response['embedding']

# Compare UniXcoder vs Generic for one operator change
original_unix = embedder.encode(original).reshape(1, -1)
one_op_unix = embedder.encode(test_cases["one_operator"]["code"]).reshape(1, -1)

original_gen = [get_ollama_embedding(original)]
one_op_gen = [get_ollama_embedding(test_cases["one_operator"]["code"])]

unix_sim = cosine_similarity(original_unix, one_op_unix)[0][0]
gen_sim = cosine_similarity(original_gen, one_op_gen)[0][0]

print("\n" + "="*70)
print("UniXcoder vs Generic Embedding Comparison")
print("="*70)
print(f"UniXcoder similarity (< → >): {unix_sim:.4f}")
print(f"Generic similarity (< → >):   {gen_sim:.4f}")
print(f"Difference:                   {abs(unix_sim - gen_sim):.4f}")
print("\nWhich model better recognizes the logical change?")
```

---

## 2.6 Variable Names Can Have Semantic Meaning
 
Create `names.py`:

```python
from sklearn.metrics.pairwise import cosine_similarity

from unixcoder_intro import UniXcoderEmbedder

embedder = UniXcoderEmbedder()

print("="*70)
print("Why Variable Names Matter to UniXcoder")
print("="*70)

# Test: Same logic, different variable name lengths
test_cases = {
    "original": "def check(x):\n    return x < 10",
    "short_rename": "def check(y):\n    return y < 10",
    "long_rename": "def check(value):\n    return value < 10",
    "very_long_rename": "def check(input_value):\n    return input_value < 10",
    "semantic_rename": "def check(threshold):\n    return threshold < 10",
}

print("\nOriginal: def check(x): return x < 10")
print("\nComparing variable name changes:")
print("-"*70)

base = embedder.encode(test_cases["original"]).reshape(1, -1)

for name, code in test_cases.items():
    if name != "original":
        emb = embedder.encode(code).reshape(1, -1)
        sim = cosine_similarity(base, emb)[0][0]
        var_name = code.split('(')[1].split(')')[0]
        print(f"{name:20} (var={var_name:15}) similarity: {sim:.4f}")

print("\n" + "="*70)
print("Insight: Variable Names Carry Semantic Information!")
print("="*70)
print("""
UniXcoder doesn't just see variables as placeholders. It recognizes:
- 'age' suggests age-related logic
- 'user_age' suggests user-specific age handling
- 'threshold' suggests boundary checking
- 'x' is generic

When you change variable names, you're potentially changing the
*semantic context* of the code, which UniXcoder picks up on!
""")
```

---


## Part 3: Building a Code Vector Store (10 minutes)

### 3.1 Introduction to Chroma with UniXcoder

Create `chroma_intro.py`:

```python
import chromadb

from unixcoder_intro import UniXcoderEmbedder

# Initialize
print("Initializing Chroma and UniXcoder...")
client = chromadb.Client()
embedder = UniXcoderEmbedder()

# Create collection
collection = client.create_collection(
    name="code_snippets_test",
    metadata={"description": "Code snippets with UniXcoder embeddings"}
)

# Sample code functions
code_snippets = [
    {
        'code': """def validate_email(email):
    '''Check if email format is valid.'''
    return '@' in email and '.' in email.split('@')[1]""",
        'name': 'validate_email',
        'purpose': 'email validation'
    },
    {
        'code': """def validate_password(password):
    '''Check if password meets security requirements.'''
    return len(password) >= 8 and any(c.isupper() for c in password)""",
        'name': 'validate_password',
        'purpose': 'password validation'
    },
    {
        'code': """def hash_password(password):
    '''Hash password using SHA-256.'''
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()""",
        'name': 'hash_password',
        'purpose': 'password hashing'
    },
    {
        'code': """def calculate_discount(price, rate):
    '''Apply discount rate to price.'''
    return price * (1 - rate)""",
        'name': 'calculate_discount',
        'purpose': 'price calculation'
    }
]

print("\n" + "="*60)
print("Adding Code to Vector Store")
print("="*60)

# Add to collection
for i, snippet in enumerate(code_snippets):
    embedding = embedder.encode(snippet['code']).tolist()
    collection.add(
        embeddings=[embedding],
        documents=[snippet['code']],
        ids=[f"code_{i}"],
        metadatas=[{
            'function_name': snippet['name'],
            'purpose': snippet['purpose']
        }]
    )
    print(f"Added: {snippet['name']}")

# Test queries
queries = [
    "How do I check if an email is valid?",
    "How do I securely store passwords?",
    "How do I apply a discount to a price?"
]

print("\n" + "="*60)
print("Querying Vector Store")
print("="*60)

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"{i}. {meta['function_name']} (distance: {dist:.4f})")
        print(f"   Purpose: {meta['purpose']}")
```

**Task 3.1:** Run and observe how Chroma works with UniXcoder.

---

### 3.2 Create a Code Knowledge Base

Create `code_kb.py`:

```python
import chromadb

from unixcoder_intro import UniXcoderEmbedder

class CodeKnowledgeBase:
    """Knowledge base for code using UniXcoder and Chroma."""
    
    def __init__(self, collection_name="code_kb"):
        print("Initializing Code Knowledge Base...")
        self.client = chromadb.Client()
        self.embedder = UniXcoderEmbedder()
        
        # Create or reset collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Code KB with UniXcoder embeddings"}
        )
        print(f"Created collection: {collection_name}")
        self.counter = 0
    
    def _create_searchable_text(self, func_name, func_code, docstring=None):
        """Create rich text representation for embedding."""
        parts = [f"Function name: {func_name}"]
        if docstring:
            parts.append(f"Description: {docstring}")
        parts.append(f"Implementation:\n{func_code}")
        return "\n".join(parts)
    
    def add_function(self, func_name, func_code, docstring=None, metadata=None):
        """Add a function to the knowledge base."""
        # Create searchable text with context
        searchable = self._create_searchable_text(func_name, func_code, docstring)
        
        # Generate embedding
        embedding = self.embedder.encode(searchable).tolist()
        
        # Prepare metadata
        meta = {
            'name': func_name,
            'docstring': docstring or '',
            'type': 'function'
        }
        if metadata:
            meta.update(metadata)
        
        # Add to collection
        doc_id = f"func_{self.counter}"
        self.collection.add(
            embeddings=[embedding],
            documents=[func_code],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        self.counter += 1
        print(f"Added: {func_name}")
    
    def search(self, query, top_k=3):
        """Search for relevant code snippets."""
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'code': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted
    
    def get_count(self):
        """Get number of items in KB."""
        return self.collection.count()

# Test
if __name__ == "__main__":
    kb = CodeKnowledgeBase("test_kb")
    
    print("\n" + "="*60)
    print("Building Code Knowledge Base")
    print("="*60 + "\n")
    
    # Add various functions
    kb.add_function(
        "validate_email",
        """def validate_email(email):
    return '@' in email and '.' in email.split('@')[1]""",
        "Check if email format is valid"
    )
    
    kb.add_function(
        "validate_password",
        """def validate_password(pwd):
    return len(pwd) >= 8 and any(c.isdigit() for c in pwd)""",
        "Check if password meets requirements"
    )
    
    kb.add_function(
        "hash_password",
        """def hash_password(pwd):
    import hashlib
    return hashlib.sha256(pwd.encode()).hexdigest()""",
        "Hash password for secure storage"
    )
    
    kb.add_function(
        "send_email",
        """def send_email(to, subject, body):
    print(f"Sending to {to}: {subject}")
    return True""",
        "Send email to recipient"
    )
    
    print(f"\nTotal functions: {kb.get_count()}")
    
    # Test search with different query types
    queries = [
        "How do I validate user email addresses?",
        "email format validation",
        "check if email is correct format"
    ]
    
    for query in queries:
        print(f"\n" + "="*60)
        print(f"Search: {query}")
        print("="*60 + "\n")
        
        results = kb.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['metadata']['name']}")
            print(f"   {result['metadata']['docstring']}")
            print(f"   Distance: {result['distance']:.4f}")
            print()
```

**Task 3.2:** Run and explore the code KB.

**Reflections:**
1. Do different phrasings of the same question retrieve the same functions?

---

## Part 4: Parsing and Indexing a Codebase (10 minutes)

### 4.1 Create Sample Codebase

Create `sample_codebase.py`:

```python
"""
Sample codebase with multiple related functions.
This simulates a small web application backend.
"""

# User Management Functions
def create_user(username, email, password):
    """Create a new user account with validation."""
    if not validate_email(email):
        return None
    if not validate_password(password):
        return None
    
    user = {
        'username': username,
        'email': email,
        'password': hash_password(password),
        'created_at': get_timestamp()
    }
    return user

def validate_email(email):
    """Validate email format has @ and domain."""
    return '@' in email and '.' in email.split('@')[1]

def validate_password(password):
    """Check password meets security requirements: 8+ chars, uppercase, digit."""
    if len(password) < 8:
        return False
    has_digit = any(c.isdigit() for c in password)
    has_upper = any(c.isupper() for c in password)
    return has_digit and has_upper

def hash_password(password):
    """Hash password using SHA-256 for secure storage."""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

# Database Functions
def save_to_database(table, data):
    """Save data record to specified database table."""
    print(f"Saving to {table}: {data}")
    return True

def query_database(table, conditions):
    """Query database table with filter conditions."""
    print(f"Querying {table} with {conditions}")
    return []

def update_record(table, record_id, updates):
    """Update existing database record with new values."""
    print(f"Updating {table} record {record_id}")
    return True

def delete_record(table, record_id):
    """Delete record from database table."""
    print(f"Deleting from {table} record {record_id}")
    return True

# Utility Functions
def get_timestamp():
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()

def send_notification(user_id, message):
    """Send push notification to user."""
    print(f"Notifying user {user_id}: {message}")
    return True

def log_event(event_type, details):
    """Log system event with timestamp."""
    timestamp = get_timestamp()
    print(f"[{timestamp}] {event_type}: {details}")

def format_error_message(error_code, context):
    """Format user-friendly error message."""
    messages = {
        'INVALID_EMAIL': 'Please enter a valid email address',
        'WEAK_PASSWORD': 'Password must be at least 8 characters',
        'USER_EXISTS': 'This username is already taken'
    }
    return messages.get(error_code, 'An error occurred')

# Authentication Functions
def authenticate_user(username, password):
    """Authenticate user credentials against database."""
    user = query_database('users', {'username': username})
    if not user:
        return None
    
    hashed = hash_password(password)
    if user[0]['password'] == hashed:
        log_event('LOGIN_SUCCESS', f'User {username} logged in')
        return user[0]
    
    log_event('LOGIN_FAILED', f'Failed login attempt for {username}')
    return None

def generate_session_token(user_id):
    """Generate random session token for authenticated user."""
    import random
    import string
    token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    save_to_database('sessions', {'user_id': user_id, 'token': token})
    return token

def validate_session(token):
    """Validate session token exists and is active."""
    session = query_database('sessions', {'token': token})
    return session is not None and len(session) > 0

def logout_user(token):
    """Invalidate user session token."""
    delete_record('sessions', token)
    return True
```

Now create `index_codebase.py`:

```python
import ast

from code_kb import CodeKnowledgeBase

def parse_python_file(filepath):
    """Parse Python file and extract functions using AST."""
    with open(filepath, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            func_lines = code.split('\n')[node.lineno-1:node.end_lineno]
            func_code = '\n'.join(func_lines)
            
            functions.append({
                'name': node.name,
                'code': func_code,
                'docstring': docstring,
                'lineno': node.lineno,
                'args': [arg.arg for arg in node.args.args]
            })
    
    return functions

def index_codebase(kb, filepath='sample_codebase.py'):
    """Index the sample codebase into knowledge base."""
    print("\n" + "="*60)
    print("Indexing Codebase with AST Parsing")
    print("="*60 + "\n")
    
    functions = parse_python_file(filepath)
    
    for func in functions:
        kb.add_function(
            func['name'],
            func['code'],
            func['docstring'],
            metadata={
                'lineno': func['lineno'],
                'args': str(func['args']),
                'file': filepath
            }
        )
    
    print(f"\nIndexed {len(functions)} functions")
    return kb

if __name__ == "__main__":
    kb = CodeKnowledgeBase("codebase_index")
    index_codebase(kb)
    
    # Test with various natural language queries
    queries = [
        "How do I validate user input?",
        "How do I authenticate a user?",
        "How do I save data to the database?",
        "How do I generate a secure token?",
        "How do I log out a user?",
        "How do I format error messages?"
    ]
    
    print("\n" + "="*60)
    print("Testing Code Search with Natural Language")
    print("="*60)
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        print("-" * 60)
        results = kb.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata']['name']}")
            print(f"   {result['metadata']['docstring']}")
            print(f"   Distance: {result['distance']:.4f}")
```

**Task 4.1:** Run and observe AST-based indexing.

---

### 4.2 Test Different Query Types

Create `test_queries.py`:

```python
from code_kb import CodeKnowledgeBase
from index_codebase import index_codebase

kb = CodeKnowledgeBase("query_test")
index_codebase(kb)

# Test various query formulations
query_tests = {
    "Natural Language": [
        "How do I check if an email is valid?",
        "What's the best way to store passwords securely?",
        "How can I verify a user's identity?",
        "How do I record what happens in the system?"
    ],
    "Technical Terms": [
        "email validation function",
        "password hashing algorithm",
        "user authentication logic",
        "database CRUD operations"
    ],
    "Use Cases": [
        "I need to create a new user account",
        "I want to log in a user",
        "I need to verify session is valid",
        "I want to save user data"
    ],
    "Incomplete/Vague": [
        "email",
        "password",
        "database",
        "user"
    ]
}

print("="*60)
print("Testing Query Types with UniXcoder")
print("="*60)

for query_type, query_list in query_tests.items():
    print(f"\n\n{'='*60}")
    print(f"Query Type: {query_type}")
    print('='*60)
    
    for query in query_list:
        print(f"\n\nQuery: '{query}'")
        print("-" * 60)
        
        results = kb.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['metadata']['name']:25} (distance: {result['distance']:.3f})")

print("\n\n" + "="*60)
print("Summary")
print("="*60)
print("Observe which query types work best:")
print("- Natural language queries")
print("- Technical terminology")
print("- Use case descriptions")
print("- Single keywords")
```

**Task 4.2:** Run and analyze retrieval across query types.

**Reflections:**
1. Which query types work best with UniXcoder?
2. How does UniXcoder handle vague vs specific queries?

---

## Part 5: Building Code RAG System (15 minutes)

### 5.1 Implement Code RAG

Create `code_rag.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from code_kb import CodeKnowledgeBase
from util import call_ollama

class CodeRAG:
    """RAG system for code generation and understanding."""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def generate_code(self, task_description, top_k=3):
        """Generate code based on task description using retrieved examples."""
        # Retrieve relevant examples
        results = self.kb.search(task_description, top_k=top_k)
        
        # Build context
        context_parts = ["Here are relevant code examples from the codebase:\n"]
        
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            context_parts.append(f"\nExample {i}: {meta['name']}")
            if meta['docstring']:
                context_parts.append(f"Purpose: {meta['docstring']}")
            context_parts.append(f"Implementation:\n{result['code']}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a code generation assistant. Based on the example code from the codebase, generate a new function.

{context}

Task: {task_description}

Requirements:
1. Follow the coding style from the examples above
2. Include a clear docstring
3. Use similar naming conventions
4. Keep the function focused and simple

Generated Function:"""
        
        # Generate
        code = call_ollama(prompt, temperature=0.3, num_predict=400)
        
        return {
            'code': code,
            'examples': results
        }
    
    def explain_code(self, code_snippet, top_k=2):
        """Explain code using similar examples from codebase."""
        # Find similar code
        results = self.kb.search(code_snippet, top_k=top_k)
        
        # Build context
        context_parts = ["Similar functions from the codebase:\n"]
        
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            context_parts.append(f"\n{i}. Function: {meta['name']}")
            context_parts.append(f"   Purpose: {meta['docstring']}")
            context_parts.append(f"   Code:\n{result['code']}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on these similar functions from the codebase:

{context}

Explain what this code does and how it works:

{code_snippet}

Provide a clear explanation covering:
1. What the code does (purpose)
2. How it works (logic)
3. How it relates to the similar examples above

Explanation:"""
        
        explanation = call_ollama(prompt, temperature=0.2, num_predict=300)
        
        return {
            'explanation': explanation,
            'similar_functions': results
        }
    
    def refactor_suggestion(self, code_snippet, top_k=3):
        """Suggest refactoring based on codebase patterns."""
        results = self.kb.search(code_snippet, top_k=top_k)
        
        context_parts = ["Best practices from the codebase:\n"]
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            context_parts.append(f"\n{i}. {meta['name']}")
            context_parts.append(f"   {meta['docstring']}")
            context_parts.append(f"   Pattern:\n{result['code']}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on these coding patterns from our codebase:

{context}

Review this code and suggest improvements:

{code_snippet}

Provide:
1. What could be improved
2. How to make it match our codebase patterns
3. Suggested refactored version

Analysis:"""
        
        suggestions = call_ollama(prompt, temperature=0.3, num_predict=400)
        
        return {
            'suggestions': suggestions,
            'reference_patterns': results
        }

if __name__ == "__main__":
    from index_codebase import index_codebase
    
    print("Building Knowledge Base...")
    kb = CodeKnowledgeBase("rag_test")
    index_codebase(kb)
    
    rag = CodeRAG(kb)
    
    # Test 1: Code Generation
    print("\n" + "="*60)
    print("Test 1: Code Generation")
    print("="*60)
    
    task = "Create a function to validate username: 3-20 characters, alphanumeric only, no spaces"
    print(f"\nTask: {task}\n")
    
    result = rag.generate_code(task)
    print("Generated Code:")
    print(result['code'])
    
    print("\n\nBased on these examples:")
    for ex in result['examples']:
        print(f"  - {ex['metadata']['name']} (distance: {ex['distance']:.3f})")
    
    # Test 2: Code Explanation
    print("\n\n" + "="*60)
    print("Test 2: Code Explanation")
    print("="*60)
    
    mystery_code = """def check_user(name, pwd):
    if len(name) < 3:
        return False
    import hashlib
    h = hashlib.sha256(pwd.encode()).hexdigest()
    return True"""
    
    print(f"\nMystery Code:\n{mystery_code}\n")
    
    result = rag.explain_code(mystery_code)
    print("Explanation:")
    print(result['explanation'])
```

**Task 5.1:** Run and examine the RAG system capabilities.

---

### 5.2 Code Understanding with RAG

Create `understand_code.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from code_kb import CodeKnowledgeBase
from index_codebase import index_codebase
from code_rag import CodeRAG

print("Initializing system...")
kb = CodeKnowledgeBase("understand_test")
index_codebase(kb)
rag = CodeRAG(kb)

# Test understanding various code snippets
test_cases = [
    {
        'name': 'Registration Function',
        'code': """def register_new_user(username, email, pwd):
    '''Register new user in system.'''
    if '@' not in email:
        return {'error': 'Invalid email'}
    if len(pwd) < 8:
        return {'error': 'Password too short'}
    
    import hashlib
    hashed = hashlib.sha256(pwd.encode()).hexdigest()
    
    user_data = {
        'username': username,
        'email': email,
        'password': hashed
    }
    
    return user_data"""
    },
    {
        'name': 'Session Check',
        'code': """def verify_user_session(token):
    sessions = get_active_sessions()
    for sess in sessions:
        if sess['token'] == token and sess['expires'] > now():
            return sess['user_id']
    return None"""
    },
]

print("\n" + "="*60)
print("Code Understanding Tests")
print("="*60)

for test in test_cases:
    print(f"\n\n{'='*60}")
    print(f"Test: {test['name']}")
    print("="*60)
    print(f"\nCode:\n{test['code']}\n")
    print("-" * 60)
    
    result = rag.explain_code(test['code'])
    
    print("Explanation:")
    print(result['explanation'])
    
    print("\n\nSimilar functions in codebase:")
    for func in result['similar_functions']:
        meta = func['metadata']
        print(f"\n  - {meta['name']} (distance: {func['distance']:.3f})")
        print(f"    {meta['docstring']}")
```

**Task 5.2:** Run and evaluate the code understanding quality.


---


## Part 6: Documentation (10 minutes)

### 6.1 Create Results Document

Create `lab7_results.md`:

```markdown
# Names: Your names here
# Lab: lab7 (RAG for Code with UniXcoder)
# Date: Today's date

---

## Submission

Submit to Git:

1. **lab7/** folder with:
   - **lab7_results.md** with **all** reflections
   - All `.py` files you created

Push to Github `cs450` repo.

Email link to `chike.abuah@wallawalla.edu`

Subject: `CS450 Lab 7`

Done!