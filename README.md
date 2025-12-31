# PrepBot 📚

An AI-powered study assistant that generates questions from your study materials and helps you learn through interactive practice.

## Features

- 📄 **Document Processing**: Upload PDFs or paste text directly
- 🧠 **Smart Topic Extraction**: Automatically identifies key concepts
- 🎯 **Adaptive Question Generation**: Questions adjust to your performance
- ✅ **Semantic Answer Evaluation**: Not just keyword matching
- 💡 **Helpful Explanations**: Learn from your mistakes
- 📊 **Progress Tracking**: See your improvement over time

## Tech Stack

- **LangChain**: Orchestration and chains
- **AWS Bedrock**: LLM (Amazon Nova Lite) and Embeddings (Titan V2)
- **ChromaDB**: Vector storage for RAG
- **SQLite**: Session and progress tracking
- **LangSmith**: Tracing and evaluation
- **PDO Prompting**: Persona-Directive-Output framework

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
source ./venv/Scripts/activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your AWS and LangSmith credentials:

### 3. Run the Application

```bash
python ./src/main.py
```

## How to Use

### 1. Create a Study Session

```
1. Create New Study Session
Enter session name: Machine Learning Midterm
✓ Session created successfully!
```

### 2. Upload Study Materials

**Option A: Upload a File**

```
3. Upload Document
1. Upload PDF or Text file
Enter file path: C:\Users\ACER\Desktop\ml_notes.pdf
✓ Document uploaded successfully!
Topics extracted: 8
```

**Option B: Paste Text**

```
3. Upload Document
2. Paste text directly
Start pasting:
[Paste your notes here]
END
✓ Text uploaded successfully!
```

### 3. Start Practicing

```
4. Start Practice Session
How many questions would you like to practice?

Question 1/5:
What is the backpropagation algorithm used for in neural networks?

Your answer: It's used to train neural networks by calculating...

✓ Correct!
Score: 85%
Feedback: Great explanation! You correctly identified...
```

### 4. Track Progress

```
5. View Progress & Analytics

Session Statistics:
  Questions Asked: 15
  Accuracy: 73.3%
  Average Score: 78.5%

Topics Needing Improvement:
  • Gradient Descent: 60%
  • Overfitting Prevention: 65%
```

## Document Upload Tips

### For File Upload:

1. **Use full paths:**

   - Windows: `C:\Users\YourName\Desktop\notes.pdf`
   - Linux/Mac: `/home/username/documents/notes.pdf`

2. **Supported formats:**

   - PDF files (`.pdf`)
   - Text files (`.txt`)
   - Markdown files (`.md`)

3. **File location tips:**
   - Put files in an easy-to-reach folder
   - Or use the `data/uploads/` folder in the project

### For Text Paste:

1. Copy your text from anywhere (web, documents, etc.)
2. Paste into the terminal
3. Type `END` on a new line when finished

## Project Structure

```
question-asker-bot/
├── src/
│   ├── config.py              # Configuration management
│   ├── bedrock_models.py      # AWS Bedrock setup
│   ├── document_processor.py  # PDF/text processing
│   ├── vector_store.py        # ChromaDB vector storage
│   ├── session_manager.py     # SQLite session tracking
│   ├── chains.py              # LangChain chains (PDO prompting)
│   ├── question_engine.py     # Main orchestration
│   └── cli.py                 # Command-line interface
├── data/
│   ├── uploads/               # Store your documents here
│   └── vector_db/             # Vector database storage
├── database/
│   └── sessions.db            # SQLite database
├── main.py                    # Run this to start!
├── requirements.txt
├── .env                       # Your credentials (create this)
└── README.md
```

## Architecture

### RAG (Retrieval-Augmented Generation)

```
Your Document → Chunks → Embeddings → Vector DB
                                         ↓
User Question → Retrieve Relevant Chunks ↓
                                         ↓
                        LLM + Context → Generate Question/Answer
```

### PDO Prompting Framework

Every LLM interaction follows:

- **Persona**: "You are an expert educator..."
- **Directive**: "Generate a question that tests..."
- **Output**: "Return JSON with: {question, topic, difficulty}"

### Adaptive Learning

```
1. Extract topics from documents
2. Track coverage: which topics asked least?
3. Track performance: which topics scored lowest?
4. Select next question: 70% coverage + 30% weak areas
5. Adjust difficulty based on accuracy
```

## Advanced Features

### Session Management

Sessions are persistent! You can:

- Close the app and resume later
- Switch between multiple study sessions
- View historical performance

## Troubleshooting

### "File not found" error

- Use absolute paths: `C:\full\path\to\file.pdf`
- Check file exists: `dir C:\path\to\file.pdf` (Windows) or `ls /path/to/file` (Linux/Mac)

### AWS Bedrock errors

- Verify credentials in `.env`
- Check model access in AWS Console → Bedrock → Model access
- Ensure region is correct (try `us-east-1`)

### "No module named" errors

- Make sure virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

## Future Enhancements

- [ ] Web interface
- [ ] Multiple choice questions
- [ ] Collaborative study sessions

## License

MIT License - feel free to use and modify!

## Contributing

Built as a learning project using:

- LangChain for orchestration
- AWS Bedrock for AI capabilities
- RAG architecture for grounded responses
- PDO prompting for consistent outputs

---

**Happy Studying! 📚✨**
