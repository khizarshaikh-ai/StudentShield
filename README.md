# StudentShield 2.0
### ML Privacy & LLM Security Research Lab

StudentShield is a local, research-oriented application for exploring privacy risks in machine learning and security weaknesses in LLM-powered applications.

Built with **Python, Streamlit, scikit-learn, and Ollama**, it provides two independent workspaces for controlled security experiments.

> **Research prototype:** StudentShield is intended for education and authorized security research. Its experimental results should not be treated as proof of real-world model security or training-data membership.

## Research Workspaces

### 1. Tabular ML Membership Inference

Train a target model on an authorized tabular dataset and evaluate whether an attacker can distinguish training records from non-training records.

Features:
- Random Forest and optional XGBoost models.
- Regression and grade-band classification.
- Target model training and member/non-member audit splits.
- Shadow-model training.
- Gaussian loss-based LiRA-style membership inference.
- ROC curves, confusion matrices, and attack metrics.
- Record-level predictions and audit results.
- CSV exports and Markdown experiment reports.

**Important:** This is a Gaussian per-record-loss LiRA-style implementation, not a complete reproduction of every canonical LiRA variant.

### 2. LLM Security Lab

Evaluate authorized local models and compatible API providers using controlled test cases.

| Experiment | Purpose |
|---|---|
| Prompt Injection | Test whether untrusted input can override system instructions. |
| Jailbreak Resistance | Evaluate responses to attempts to bypass a benign safety boundary. |
| RAG Canary Leakage | Check whether a synthetic confidential token is repeated by the model. |
| Experimental LLM MIA | Explore likelihood-based membership signals when token log-probabilities are supported. |

The application supports:
- Local Ollama models.
- OpenAI-compatible chat APIs.
- Editable system and attack prompts.
- Saved prompt test cases within the current session.
- Model response inspection.
- Synthetic canary generation.
- CSV export of experiment results.

**Important research distinctions:**
- Tabular membership inference and LLM security testing are separate experiments.
- A synthetic RAG canary test does not constitute a complete vector database or multi-user authorization audit.
- The LLM membership probe is an experimental average-token-log-probability baseline, not full LiRA.
- Prompt detection and jailbreak outcomes may require manual review.
- A successful or unsuccessful individual test does not establish general model robustness.

## Technology Stack

- **Language:** Python
- **Interface:** Streamlit
- **Data processing:** NumPy, Pandas
- **Machine learning:** scikit-learn, optional XGBoost
- **Visualization:** Matplotlib
- **LLM integration:** Ollama and OpenAI-compatible APIs
- **HTTP communication:** Requests

## Project Structure

```text
StudentShield/
├── app.py
├── start.bat
├── requirements.txt
├── README.md
├── .gitignore
└── studentshield/
    ├── __init__.py
    ├── core.py
    ├── attacks.py
    ├── reporting.py
    └── llm_security.py
```

## Installation — Windows

### Requirements

- Windows 10 or 11
- Python 3.11 or 3.12 (64-bit recommended)
- Internet connection for initial dependency installation
- Ollama, if you want to run local LLM experiments

### 1. Clone the repository

```powershell
git clone https://github.com/YOUR_USERNAME/StudentShield.git
cd StudentShield
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start StudentShield
