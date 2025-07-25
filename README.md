# DLVAIA: Deliberately Left Vulnerable AI Application 🧪

[![GitHub license](https://img.shields.io/github/license/Iam-M-i-r-z-a/DLVAIA?style=for-the-badge)](https://github.com/Iam-M-i-r-z-a/DLVAIA/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama)](https://ollama.com/)

---
## ⚠️ Caution

> **This project is deliberately insecure.**
>
> - ❌ **Do NOT deploy this application in production.**
> - ❌ **NEVER enter real personal, sensitive, or confidential data.**
>
> DLVAIA is intended for **educational and research purposes only** to explore and demonstrate AI/LLM security vulnerabilities. Misusing it may cause harm or violate privacy policies.



---

## 🌟 About This Project

Welcome to **DLVAIA (Deliberately Left Vulnerable AI Application)**!

This project is a hands-on learning environment designed to help you understand and identify critical security vulnerabilities in Large Language Models (LLMs). Inspired by the **OWASP Top 10 for LLMs**, DLVAIA provides a practical sandbox to explore common attack vectors and defense mechanisms.

**A key advantage of DLVAIA is its 100% local deployability.** Due to Ollama, you can run the entire application, including the LLM inference, completely on your own machine. This means:
* **No API Keys Required:** You won't need to connect to remote services like Claude, ChatGPT, or Gemini, eliminating dependency on third-party APIs.
* **Enhanced Privacy:** Your data stays on your machine, ensuring maximum privacy and control over your interactions.
* **Cost-Free Inference:** There are no recurring costs associated with LLM usage, making it ideal for continuous learning and experimentation.

Whether you're a security researcher, a developer building with LLMs, or an enthusiast curious about AI security, DLVAIA offers a unique opportunity to test your understanding in a controlled, private, and educational setting.

---



## 🎯 Challenges You'll Explore

DLVAIA specifically features and allows you to experiment with the following OWASP Top 10 LLM vulnerabilities:

* **Prompt Injection:** Manipulating the model's output through crafted inputs.
* **Insecure Output Handling:** Exploiting how the application processes and displays LLM responses.
* **Training Data Poisoning:** Understanding the impact of compromised training data.
* **Model Denial of Service:** Disrupting the LLM's availability or performance.
* **Supply Chain Vulnerabilities:** Identifying risks in the components and integrations used.
* **Data Leakage (Sensitive Data Disclosure):** Discovering unintended exposure of confidential information.
* **Model Theft:** Exploring methods to extract or misuse proprietary models.
* **Overreliance:** Recognizing the dangers of excessive trust in LLM outputs.
* **Excessive Agency:** Examining risks when LLMs are granted too much autonomy.
* **Insecure Plugins:** Understanding vulnerabilities introduced by external tools or plugins.

---

## 🚀 Getting Started

Follow these steps to set up and run DLVAIA on your local machine.

### Prerequisites

Make sure you have `git` and `Python 3.x` installed on your system.

### Ollama Setup (Local LLM)

DLVAIA leverages [Ollama](https://ollama.com/) for running local LLMs, ensuring your data stays private and you have full control over the model.

1.  **Download Ollama:**
    Visit the official Ollama website and download the installer for your operating system:
    ➡️ [**ollama.com/download**](https://ollama.com/download)

2.  **Install an LLM Model:**
    After installing Ollama, open your terminal or command prompt and pull the `mistral:instruct` model. This is the default model used by DLVAIA.
    ```bash
    ollama pull mistral:instruct
    ```
    *(**Note:** You can choose a different model if you prefer, but remember to update the model name within the project's code accordingly.)*

### Deploying the Application

With Ollama ready, you can now deploy DLVAIA:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Iam-M-i-r-z-a/DLVAIA.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd DLVAIA
    ```

3.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    * **Linux / macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate
        ```
    * **Windows (PowerShell):**
        ```bash
        .\venv\Scripts\Activate.ps1
        ```

5.  **Install Dependencies:**
    Install all required Python packages using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the Application:**
    Start the Flask application.
    ```bash
    python app.py
    ```

7.  **Access DLVAIA:**
    Open your web browser and visit the application:
    [http://localhost:5000/](http://localhost:5000/)

---

## 🤝 Contributions & Feedback

Contributions, bug reports, and feature requests are welcome! If you find a new vulnerability pattern or have ideas for improving DLVAIA, please open an issue or submit a pull request.

---

## ✉️ Connect with Me

* **GitHub:** [Iam-M-i-r-z-a](https://github.com/Iam-M-i-r-z-a)
* **Linkedin:** [abdelrahman-hesham-b208b427b](https://www.linkedin.com/in/abdelrahman-hesham-b208b427b/)

---
