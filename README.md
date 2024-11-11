# Al-Enhanced Content Creation for University Courses
 In today's digital education landscape, the need for original and high-quality content is critical. This project leverages AI to enhance the creation of university course materials, focusing on generating content that is both engaging and plagiarism-free. By integrating advanced AI models, we aim to streamline the content development process for educators
![image](https://github.com/user-attachments/assets/864b58f4-5827-4c5c-9dc1-a0e12c3696c4)
## Table of Contents
- [Business Objective](#business-objective)
- [Project Scope](#project-scope)
- [Technical Stack](#technical-stack)
- [Modeling & Architecture](#modeling--architecture)
- [Deployment](#deployment)
- [Getting Started](#getting-started)
- [Contribute](#Contribute)
## Business Objective
The primary objective is to reduce the time and effort needed to generate and retrieve relevant course content. Key goals include:

- Efficiency: Minimizing time for content search and creation.
- Accuracy: Ensuring AI understands and responds accurately based on context.
- Plagiarism-Free: Providing unique responses derived from document context or general knowledge.
## Project Scope
- Software: Visual Studio Code
- Libraries: FAISS, Google Generative AI (genAI), LangChain, PyPDF2, Streamlit
- System Requirements:
 - Hardware: Minimum 4GB RAM, 50GB storage, Intel i5 (8th gen or higher)
 - Operating System: Windows 10/11 or Linux (Ubuntu)
 - Network: High-speed internet
## Technical Stack
- **Software:** Visual Studio Code
- **Libraries:** FAISS, Google Generative AI (genAI), LangChain, PyPDF2, Streamlit
- **System Requirements:**
    - **Hardware:** Minimum 4GB RAM, 50GB storage, Intel i5 (8th gen or higher)
    - **Operating System:** Windows 10/11 or Linux (Ubuntu)
    - **Network:** High-speed internet
## Modeling & Architecture
1. **Data Processing:** Text extraction and chunking from uploaded PDFs.
2. **Embedding Generation:** Text chunks vectorized with Google Generative AI embeddings.
3. **Question Answering:** Answers generated based on context or general knowledge using LangChain’s QA Chain.
![image](https://github.com/user-attachments/assets/0fd7bdc6-3f03-403a-8702-180a8f474328)
![image](https://github.com/user-attachments/assets/b220a069-c525-4def-970f-0f0db1efb0c9)

## Deployment
The application leverages the following deployment strategies:

- Model Selection: Uses Google Generative AI (Gemini API) for advanced NLP capabilities.
- Embedding & Retrieval Strategy: Embeddings stored in FAISS for fast similarity-based searches.
- Plagiarism Mitigation: Ensures content originality through model fine-tuning.
## Getting Started
1. **Clone Repository:**  
   ```bash
   git clone <repository-url>
2. **Install Dependencies:**
Use requirements.txt for installation.
  ```bash
pip install -r requirements.txt
```
3. **Run Application:**
Launch the Streamlit application.
```bash
  streamlit run app.py
```
## Contribute
- Gaurav
