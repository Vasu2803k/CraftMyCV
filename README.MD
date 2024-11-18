# CraftMyCV

An intelligent CV/resume customization system powered by AI agents that analyzes, optimizes, and tailors your resume for specific job descriptions. The system uses multiple specialized AI agents working in concert to create highly optimized, ATS-friendly resumes.

## 🌟 Features

- **Intelligent Analysis**: Multiple AI agents analyze both resumes and job descriptions
- **Multi-format Support**: 
  - Documents: PDF, DOCX, DOC, TXT
  - Images: PNG, JPG, JPEG
  - Built-in OCR capabilities for scanned documents
- **Advanced Processing**:
  - Automatic text extraction with OCR fallback
  - Skill matching and optimization
  - ATS-friendly formatting
  - Professional LaTeX output
- **Robust Architecture**:
  - Multiple LLM support (OpenAI GPT and Anthropic Claude)
  - Fallback LLM system for reliability
  - Extensive error handling
  - Configurable agent behaviors

## 🤖 AI Agents

The system employs 8 specialized agents:

1. **Resume Analyzer**: Analyzes and categorizes resume content
2. **Job Description Analyzer**: Extracts key requirements and cultural elements
3. **Skill Customizer**: Optimizes skills section for job alignment
4. **Resume Customizer**: Enhances content for job requirements
5. **Summary Customizer**: Optimizes professional summary
6. **Information Synthesizer**: Combines analyses into structured data
7. **Format Converter**: Creates LaTeX-formatted output
8. **Quality Controller**: Ensures ATS compliance and quality

## 🛠 Prerequisites

- Python 3.11+
- Tesseract OCR
- API Keys:
  - OpenAI API key
  - Claude API key (Anthropic)
  - Fallback LLM API key

## 📦 Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd craft-my-cv
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate craft-my-cv
```

3. Set up environment variables:
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
FALLBACK_LLM_API_KEY=your_fallback_api_key
```

## 🚀 Usage

Run via command line:
```bash
python src/main.py --resume path/to/resume.pdf --job-description "job description text"
```

Options:
- `--resume`: Path to resume file
- `--job-description`: Job description text or file path
- `--output`: Output path (default: output.json)

## 🏗 Project Structure
```
craft-my-cv/
├── src/
│ ├── config/
│ │ ├── agents.yaml # Agent configurations
│ │ └── tasks.yaml # Task definitions
│ ├── tools/
│ │ └── text_extractor_tool.py
│ ├── utils/
│ │ └── fallback_llm.py
│ ├── crew.py # Main agent orchestration
│ └── main.py # CLI interface
├── environment.yml # Dependencies
└── README.md
```

## 🔧 Configuration

### Agent Configuration (agents.yaml)
- Agent roles and goals
- LLM settings
- System prompts
- Fallback configurations

### Task Configuration (tasks.yaml)
- Task definitions
- Input/output schemas
- Processing instructions
- Validation rules

## 📚 Dependencies

Core dependencies include:
- **AI/ML Frameworks**:
  - crewai
  - langchain
  - openai
  - anthropic
  - transformers
  - sentence-transformers
  - pytorch

- **Document Processing**:
  - pypdf2
  - python-docx
  - pdfminer.six
  - tesseract
  - pytesseract
  - pillow

- **Utilities**:
  - numpy
  - pandas
  - python-dotenv
  - pyyaml
  - tqdm

## 🔍 Error Handling

The system includes comprehensive error handling for:
- File operations
- API failures
- Text extraction
- OCR processing
- Input validation
- LLM failures (with fallback options)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[Add license information]

## 🙏 Acknowledgments

- CrewAI framework
- OpenAI GPT models
- Anthropic Claude models
- Tesseract OCR