# Lab 5: Basic Multimodal AI with Ollama

**Lab Duration:** 50 minutes  

---

## Learning Objectives

By the end of this lab, you will be able to:
- Use Ollama's multimodal API to analyze images
- Apply vision models to image recognition and classification tasks
- Combine text and image inputs for enhanced AI interactions
- Evaluate multimodal model performance

---

## Prerequisites

- Python 
- Network Access to the course Ollama server (Via CS Lab)
- Completed Lab 1 and Lab 2

---

## Resources

- Ollama Python library docs: https://github.com/ollama/ollama-python
- Ollama Vision API: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
- Ollama Models: https://ollama.com/library

---

## Part 1: Setup (5 minutes)

Run the following commands: 

```bash
cd cs450
mkdir lab5
cd lab5
```

Add the following function to your ollama client:

```python
def analyze_image(image_path, prompt, model="gemma3", temperature=0.3):
    """
    Analyze an image with a text prompt.
    
    Args:
        image_path: Path to image file
        prompt: Question or instruction about the image
        model: Vision model to use
        temperature: Randomness (0.0-1.0)
    """
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }],
        options={'temperature': temperature}
    )
    return response['message']['content']
```

**Note:** We'll use publicly available test images from the internet for this lab.

---

## Part 2: Image Description (10 minutes)

### 2.1 Basic Image Description

Create `describe.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import analyze_image

def describe_image(image_url, detail_level="basic"):
    """Generate description of an image."""
    
    if detail_level == "basic":
        prompt = "Describe this image in one sentence."
    elif detail_level == "detailed":
        prompt = "Provide a detailed description of this image, including objects, colors, and context."
    else:
        prompt = "What do you see in this image?"
    
    response = analyze_image(image_url, prompt)
    return response

# Test with classic freely licensed images
test_images = [
    "https://upload.wikimedia.org/wikipedia/commons/3/37/Oryctolagus_cuniculus_Tasmania_2.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/5f/Red_Kangaroos_at_Sturt_National_Park_NSW.jpg"
]

if __name__ == "__main__":
    print("Image Description\n" + "="*50)
    
    for img_url in test_images:
        print(f"\nImage: {img_url}")
        
        # Basic description
        basic = describe_image(img_url, "basic")
        print(f"Basic: {basic}")
        
        # Detailed description
        detailed = describe_image(img_url, "detailed")
        print(f"Detailed: {detailed[:100]}...")
```

**Task 2.1:** Run this script and observe the different description levels.

**Reflections:**
1. How do basic vs detailed descriptions differ?
2. What details does the model focus on?

---

### 2.2 Targeted Questions

Create `question.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import analyze_image

def ask_about_image(image_url, question):
    """Ask specific questions about an image."""
    response = analyze_image(image_url, question, temperature=0.1)
    return response


test_image = "https://upload.wikimedia.org/wikipedia/commons/9/95/Man_biking_on_Recife_city.jpg"

questions = [
    "What is the main subject of this image?",
    "What colors are prominent in this image?",
    "Is this a photograph or a drawing?",
    "Describe the person's appearance."
]

if __name__ == "__main__":
    print("Question-Answering with Images\n" + "="*50)
    
    for question in questions:
        answer = ask_about_image(test_image, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
```

**Task 2.2:** Run and evaluate the answers. Try adding your own questions.

---

## Part 3: Image Classification (10 minutes)

### 3.1 Zero-Shot Image Classification

Create `classify_zero.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import analyze_image

def classify_image_zeroshot(image_url, categories):
    """Classify image into one of the given categories."""
    
    categories_str = ", ".join(categories)
    prompt = f"""Classify this image into ONE of these categories: {categories_str}
Return ONLY the category name, nothing else."""
    
    response = analyze_image(image_url, prompt, temperature=0.1)
    return response.strip()

# Test with various Wikipedia Commons images
test_cases = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg",
        "categories": ["dog", "cat", "bird", "other animal"]
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg",
        "categories": ["building", "vehicle", "nature", "animal"]
    }
]

if __name__ == "__main__":
    print("Zero-Shot Image Classification\n" + "="*50)
    
    for test in test_cases:
        result = classify_image_zeroshot(test["url"], test["categories"])
        print(f"\nImage: {test['url']}")
        print(f"Categories: {test['categories']}")
        print(f"Classification: {result}")
```

**Task 3.1:** Run this and check accuracy. Try different category sets.

---

### 3.2 Few-Shot Image Classification

Create `classify_fewshot.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import analyze_image

def classify_activity_basic(image_url):
    """Basic work vs leisure classification."""
    
    prompt = """Classify this image as WORK or LEISURE.

Return only: WORK or LEISURE"""
    
    response = analyze_image(image_url, prompt, temperature=0.1)
    return response.strip()

def classify_activity_guided(image_url):
    """Classification with specific guidelines for edge cases."""
    
    prompt = """Classify this image as WORK or LEISURE.

Guidelines:
- WORK: Any activity that could generate income or is done professionally:
  * Professional sports like running & track & field (even if enjoyable)
  * Content creation (photography, streaming)
  * Playing music (including street performers)
  * Fitness like a Yoga instructor
  * Competitive activities with prizes/sponsorships
  * Volunteer work or unpaid internships
  
- LEISURE: Activities purely for personal enjoyment:
  * Recreational sports without competition
  * Hobbies without monetization intent
  * Social gatherings
  * Personal fitness

Return only: WORK or LEISURE"""
    
    response = analyze_image(image_url, prompt, temperature=0.1)
    return response.strip()

# Ambiguous images where guidelines matter
test_images = [
    # Professional athlete training
    "https://upload.wikimedia.org/wikipedia/commons/1/19/Sergio_Ottolina_1964.jpg",
    
    # Street musician (could be work or hobby)
    "https://upload.wikimedia.org/wikipedia/commons/3/3d/Arles_Busker_IMG_8299.jpg",
    
    # Yoga class (fitness or instructor?)
    "https://upload.wikimedia.org/wikipedia/commons/8/87/Yoga_Class.jpg"
]

if __name__ == "__main__":
    print("Work vs Leisure Classification\n" + "="*50)
    
    for img_url in test_images:
        basic = classify_activity_basic(img_url)
        guided = classify_activity_guided(img_url)
        
        print(f"\nImage: {img_url.split('/')[-1]}")
        print(f"Basic classification:  {basic}")
        print(f"With guidelines:       {guided}")
        print(f"Different result:      {'YES ✓' if basic != guided else 'NO'}")
```

**Task 3.2:** Compare this to zero-shot. Does adding guidelines help? In what situations do the guidelines seem to matter most?

---

## Part 4: Image Recognition Tasks (10 minutes)

### 4.1 Object Detection

Create `detect.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import analyze_image

def detect_objects(image_url):
    """Detect and list objects in an image."""
    
    prompt = """List all objects you can identify in this image.
Format: Return a comma-separated list of objects."""
    
    response = analyze_image(image_url, prompt, temperature=0.2)
    return response

def count_objects(image_url, object_type):
    """Count specific objects in an image."""
    
    prompt = f"How many {object_type} are in this image? Return only a number."
    response = analyze_image(image_url, prompt, temperature=0.1)
    return response

if __name__ == "__main__":
    # Using a simple test image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    
    print("Object Detection\n" + "="*50)
    
    # List objects
    objects = detect_objects(image_url)
    print(f"\nDetected objects: {objects}")
    
    # Count specific objects
    count = count_objects(image_url, "circles")
    print(f"Circle count: {count}")
```

**Task 4.1:** Does the object count seem accurate? 

**Task 4.2:** Try testing with different images and object types.

---

### 4.2 Visual Question Answering

Create `visual_qa.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util import analyze_image

def visual_qa(image_url, questions):
    """Answer multiple questions about an image."""
    
    results = []
    for question in questions:
        answer = analyze_image(image_url, question, temperature=0.2)
        results.append((question, answer))
    
    return results

# Test image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons.jpg/360px-Tour_Eiffel_Wikimedia_Commons.jpg"

questions = [
    "What landmark is shown in this image?",
    "What time of day is it?",
    "Are there people visible?",
    "What's the weather like?"
]

if __name__ == "__main__":
    print("Visual Question Answering\n" + "="*50)
    
    results = visual_qa(image_url, questions)
    
    for q, a in results:
        print(f"\nQ: {q}")
        print(f"A: {a}")
```

**Task 4.2:** Run and evaluate answer quality.

---

## Part 6: Evaluation (10 minutes)

### 6.1 Document Your Findings

Create a file `lab5_results.md` in `lab5` and add:

```markdown
# Names: Your name and your lab partner's name
# Lab: Lab5 (Multimodal AI with Ollama)
# Date: Today's date
```

Now, answer **(without using GenAI)**:

**1. Image Understanding:**
- How accurate were the image descriptions?
- What types of details did the model capture well?
- What did it miss or misinterpret?

**2. Classification Performance:**
- How did zero-shot classification perform?
- Did adding guidelines (few-shot style) improve results?
- What categories were easiest/hardest to classify?

**3. Practical Applications:**
- What real-world applications could benefit from this?
- What limitations did you encounter?
- How might you improve accuracy?

**4. Comparison:**
- How does multimodal AI differ from text-only?
- What new capabilities does vision enable?

---

## Submission

Stage and commit the following files to Git:
 
1. **lab5/** folder with:
   - **lab5_results.md** with your with answers to Tasks `3.2` and `4.1`.
   - The rest of your `lab5` directory

Push all code and md files to your `cs450` GitHub repository.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

Email subject should be `CS450 LAB 5`

Great work exploring multimodal AI! 🖼️

---