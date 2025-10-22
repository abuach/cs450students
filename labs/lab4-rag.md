# Lab 3: Retrieval Augmented Generation (RAG)

**Lab Duration:** 50 minutes  

---

## Learning Objectives

By the end of this lab, you will be able to:
- Build a simple document-based knowledge base
- Implement basic semantic search using embeddings
- Combine retrieved information with LLM prompts
- Understand when RAG improves answer quality

---

## Prerequisites

- Python 
- Network Access to the course Ollama server (Via CS Lab)
- Completed Lab 3

---

## Resources

- Ollama Python library docs: https://github.com/ollama/ollama-python
- Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
- https://docs.ollama.com/ 
- https://docs.ollama.com/api#generate-embeddings

---

## Part 1: Setup (5 minutes)

Run the following commands: 

```bash
unset LD_PRELOAD
cd cs450
uv add scikit-learn
uv add PyPDF2
mkdir lab4
cd lab4
```

Remember to use `uv run [program].py` to run your code.

---

## Part 2: Understanding Embeddings (10 minutes)

### 2.1 Generate Embeddings

Create `embeddings_intro.py`:

```python
import sys
from pathlib import Path
import ollama

sys.path.append(str(Path(__file__).parent.parent))

def get_embedding(text):
    """Get embedding vector for text."""
    client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')
    response = client.embeddings(
        model='cs450',
        prompt=text
    )
    return response['embedding']

# Test embeddings
texts = [
    "The cat sat on the mat",
    "A feline rested on a rug",
    "The dog played in the park"
]

print("Generating Embeddings\n" + "="*50)
for text in texts:
    embedding = get_embedding(text)
    print(f"\nText: {text}")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
```

**Task 2.1:** Run this and observe the embedding dimensions.

**Reflections:**
1. How long are the embedding vectors?
2. Are the vectors different for each text?

---

### 2.2 Measure Similarity

Create `similarity_test.py`:

```python
import sys
from pathlib import Path
import ollama
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))

def get_embedding(text):
    """Get embedding vector for text."""
    client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')
    response = client.embeddings(
        model='cs450',
        prompt=text
    )
    return response['embedding']

# Compare similar and different texts
text1 = "The cat sat on the mat"
text2 = "A feline rested on a rug"
text3 = "Python is a programming language"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# Calculate similarities
sim_1_2 = cosine_similarity([emb1], [emb2])[0][0]
sim_1_3 = cosine_similarity([emb1], [emb3])[0][0]

print("Semantic Similarity Test\n" + "="*50)
print(f"\nText 1: {text1}")
print(f"Text 2: {text2}")
print(f"Similarity: {sim_1_2:.4f}")

print(f"\nText 1: {text1}")
print(f"Text 3: {text3}")
print(f"Similarity: {sim_1_3:.4f}")
```

**Task 2.2:** Run and observe similarity scores.

**Reflections:**
1. Which texts are more similar?
2. Does semantic similarity match your intuition?
3. Try changing the texts to test similarity between other phrases or sentences.

---

## Part 3: Building a Knowledge Base (10 minutes)

### 3.1 Simple Document Store

Create `kb.py`:

```python
import sys
from pathlib import Path
import ollama
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))

class SimpleKnowledgeBase:
    """A simple document store with semantic search."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')
    
    def add_document(self, text):
        """Add a document to the knowledge base."""
        # Get embedding
        response = self.client.embeddings(
            model='cs450',
            prompt=text
        )
        embedding = response['embedding']
        
        # Store document and embedding
        self.documents.append(text)
        self.embeddings.append(embedding)
        print(f"Added document: {text[:50]}...")
    
    def search(self, query, top_k=2):
        """Search for most relevant documents."""
        # Get query embedding
        response = self.client.embeddings(
            model='cs450',
            prompt=query
        )
        query_embedding = response['embedding']
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding], 
            self.embeddings
        )[0]
        
        # Get top-k documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': similarities[idx]
            })
        
        return results

# Test the knowledge base
if __name__ == "__main__":
    kb = SimpleKnowledgeBase()
    
    # Add documents
    documents = [
        "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "JavaScript is primarily used for web development and runs in browsers.",
        "Machine learning is a subset of AI focused on learning from data.",
        "React is a JavaScript library for building user interfaces.",
        "Python is widely used in data science and machine learning."
    ]
    
    print("Building Knowledge Base\n" + "="*50)
    for doc in documents:
        kb.add_document(doc)
    
    # Test search
    query = "What is Python used for?"
    print(f"\n\nSearching for: {query}\n" + "="*50)
    results = kb.search(query)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   {result['text']}")
```

**Task 3.1:** Run and examine search results.

**Reflections:**
1. Are the top results relevant to the query?
2. What makes a document relevant?

---

### 3.2 Add More Documents

Create `test_kb.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import your SimpleKnowledgeBase
from kb import SimpleKnowledgeBase

# Create knowledge base
kb = SimpleKnowledgeBase()

# Add CS course information
documents = [
    "CS450 focuses on software engineering and AI applications.",
    "Data structures include arrays, linked lists, trees, and graphs.",
    "Algorithms course covers sorting, searching, and dynamic programming.",
    "Web development uses HTML, CSS, JavaScript, and frameworks like React.",
    "Databases store and organize data using SQL or NoSQL systems.",
    "Operating systems manage hardware and software resources.",
    "Computer networks enable communication between devices."
]

print("Adding Documents\n" + "="*50)
for doc in documents:
    kb.add_document(doc)

# Test queries
queries = [
    "What does CS450 cover?",
    "Tell me about data structures",
    "How do databases work?"
]

print("\n\nTesting Queries\n" + "="*50)
for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 50)
    results = kb.search(query, top_k=2)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['score']:.3f}] {result['text']}")
```

**Task 3.2:** Run and evaluate retrieval quality!

---

## Part 4: RAG Implementation (10 minutes)

### 4.1 Basic RAG System

Create `simple_rag.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from kb import SimpleKnowledgeBase
from util import call_ollama

class SimpleRAG:
    """Simple RAG system combining retrieval and generation."""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def answer_question(self, question):
        """Answer question using RAG."""
        # Step 1: Retrieve relevant documents
        results = self.kb.search(question, top_k=2)
        
        # Step 2: Build context from retrieved docs
        context = "\n".join([r['text'] for r in results])
        
        # Step 3: Create prompt with context
        prompt = f"""Answer the question based on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
        
        # Step 4: Generate answer
        answer = call_ollama(prompt, temperature=0.2, num_predict=150)
        
        return {
            'answer': answer,
            'sources': results
        }

# Test RAG
if __name__ == "__main__":
    # Build knowledge base
    kb = SimpleKnowledgeBase()
    
    documents = [
        "Python was created by Guido van Rossum and released in 1991.",
        "Python emphasizes code readability with significant indentation.",
        "Python supports multiple programming paradigms including OOP.",
        "Python is used in web development, data science, and automation.",
        "JavaScript was created by Brendan Eich in 1995 for web browsers.",
        "JavaScript is the primary language for client-side web development."
    ]
    
    print("Building Knowledge Base\n" + "="*50)
    for doc in documents:
        kb.add_document(doc)
    
    # Create RAG system
    rag = SimpleRAG(kb)
    
    # Test questions
    questions = [
        "Who created Python?",
        "What is Python used for?",
        "When was JavaScript created?",
        "What is Python's execution speed?"  # Not in KB
    ]
    
    print("\n\nTesting RAG System\n" + "="*50)
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 50)
        result = rag.answer_question(question)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. [{source['score']:.3f}] {source['text'][:60]}...")
        print("="*50)
```

**Task 4.1:** Run and observe how RAG uses retrieved context.

**Reflections:**
1. Does RAG answer questions accurately?
2. What happens when information isn't in the knowledge base?
3. Are the retrieved sources relevant?

---

### 4.2 Compare with and without RAG

Create `compare_rag.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from kb import SimpleKnowledgeBase
from simple_rag import SimpleRAG
from util import call_ollama

# Build knowledge base with specific information
kb = SimpleKnowledgeBase()

documents = [
    "Walla Walla University was founded in 1892 in College Place, Washington.",
    "WWU offers undergraduate and graduate programs in various fields.",
    "The university is affiliated with the Seventh-day Adventist Church.",
    "WWU's campus is located in the Walla Walla Valley, known for wine production."
]

print("Adding Documents to Knowledge Base\n" + "="*50)
for doc in documents:
    kb.add_document(doc)

# Create RAG system
rag = SimpleRAG(kb)

# Test question
question = "When was Walla Walla University founded?"

print("\n\nComparison: With and Without RAG\n" + "="*50)

# Without RAG (just LLM)
print("\nWITHOUT RAG (Direct LLM):")
print("-" * 50)
direct_prompt = f"Answer this question: {question}"
direct_answer = call_ollama(direct_prompt, temperature=0.2, num_predict=100)
print(f"Answer: {direct_answer}")

# With RAG
print("\n\nWITH RAG (Retrieved Context + LLM):")
print("-" * 50)
rag_result = rag.answer_question(question)
print(f"Answer: {rag_result['answer']}")
print(f"\nSources:")
for i, source in enumerate(rag_result['sources'], 1):
    print(f"  {i}. {source['text'][:70]}...")
```

**Task 4.2:** Run and compare the answers.

**Reflections:**
1. Which approach gives more accurate answers?
2. When would you prefer RAG over direct LLM queries?

---

## Part 5: RAG with Large Files (10 minutes)

### Exercise 5.1: Course FAQ System

Download the course syllabus into your lab directory. Name it `syllabus.pdf`

Create `course_faq.py`:

```python
import sys
from pathlib import Path
import PyPDF2

sys.path.append(str(Path(__file__).parent.parent))

from kb import SimpleKnowledgeBase
from simple_rag import SimpleRAG

def parse_pdf_to_array(pdf_path):
    """Parse PDF pages into an array of strings."""
    pages = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page in reader.pages:
            text = page.extract_text()
            pages.append(text)
    
    return pages


def build_course_kb():
    """Build knowledge base with course information."""
    kb = SimpleKnowledgeBase()

    pdf_file = "syllabus.pdf"
    documents = parse_pdf_to_array(pdf_file)
    
    for doc in documents:
        kb.add_document(doc)
    
    return kb

def test_faq_system():
    """Test the FAQ system."""
    kb = build_course_kb()
    rag = SimpleRAG(kb)
    
    # TODO: Create test questions
    questions = [
        "What topics does CS450 cover?",
        "What are the CS450 student learning objectives?",
        "What is the CS450 Week of Worship?",
        "Tell me about the Week of Worship",
        "Tell me about the CS450 instructor",
        "Instructor:"
    ]
    
    print("Course FAQ System\n" + "="*50)
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 50)
        result = rag.answer_question(question)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. [{source['score']:.3f}] {source['text'][:60]}...")
        print("="*50)
        print()

if __name__ == "__main__":
    test_faq_system()
```

**Task 5.1:** Run this and observe the output for each question.

**Reflections:**
1. Does the structure of the questions seem to matter?
2. Would you say that RAG has limitations based on this exercise?
3. Did you notice anything about the course name in the questions? Is that course name actually in the context?
    What does this demonstrate?

---

## Part 6: Evaluation (5 minutes)

### 6.1 Document Your Findings

Create a file `lab4_results.md` in `lab4` and add:

```markdown
# Names: Your name and your lab partner's name
# Lab: Lab3 (Retrieval Augmented Generation)
# Date: Today's date
```

`Question x.y.z` is shorthand for `Reflection z` for `Task x.y` 

---

## Submission

Stage and commit the following files to Git:
 
1. **lab4/** folder with:
   - **lab4_results.md** with answers to Questions `5.1.1`, `5.1.2` and `5.1.3`.
   - All other `.py` files you created

Push to your remote Github `cs450` repo.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

Email subject should be `CS450 LAB 4`

You're done with the fourth lab! 🎯😎

---