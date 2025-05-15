# **Image Captioning with Flickr8k Dataset**

## Overview
This project builds a story continuation system that:

-**Loads predefined story slides**
-**Embeds them using HuggingFace transformer embeddings**
-**Stores embeddings in a Chroma vector store (run once)**
-**Accepts user instructions to continue the story**
-**Retrieves the most relevant slide using vector similarity and reranks with a LLaMA LLM**
-**Generates the next story slide based on context and user input**

 
 


## Project Highlights
-**Slides:** Predefined story slides to build contextual story prompts

-**Embedding:** Uses HuggingFace sentence-transformer models for embedding text

-**Vector Search:** Uses Chroma vector store for similarity search

-**LLM:** Uses LLaMA model (via Ollama or your choice) for prompt continuation

-**Modular:** Code split into modules for data loading, vector store, prompt building, and LLM interaction



## Installation

To get started with the project, follow the steps below:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kprem-2001/Context-Aware_Prompt_Builder.git
   cd  Context-Aware_Prompt_Builder

2. **Create Virtual Environemnt**:
   ```bash
   conda create -n name python==3.10 -y

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
  
# Usage 🚀
To use this project, you can either run it as a command-line tool or as a web interface.

### Command-line tool 💻
To run the project as a command-line tool, you can use the following command:
**1.Run this Command First and Only One time**
```bash
python vector_store.py
```
This will generate vector store in chroma db 

**2.Run this Command after that**

```bash
python main.py
```


# 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prem-kumar-b1aa57257/)
[![email](https://img.shields.io/badge/email-0088cc?style=for-the-badge&logo=gmail&logoColor=white)](mailto:kumarprem10694@gmail.com)