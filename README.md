# RAG ChatBot

A conversational chatbot designed for the North South University ECE Department for CSE299 junior design project. This project leverages advanced machine learning techniques using the LangChain framework and Gradio UI to create an interactive chatbot capable of answering questions about the department.

## Retrieval-Augmented Generation (RAG) Pipeline

### Overview of RAG
Retrieval-Augmented Generation (RAG) is a hybrid machine learning approach that combines the strengths of information retrieval systems and generative language models. It enhances the accuracy and relevance of responses by fetching contextually relevant information from a knowledge base before generating an answer. 

### Why RAG?
- **Improved Accuracy**: By retrieving relevant documents, the generative model is better informed, leading to more precise answers.
- **Dynamic Context**: Dynamically incorporates up-to-date or specific context from the retrieval system.
- **Scalability**: Handles large knowledge bases efficiently by leveraging embeddings and similarity searches.

### Features
- Combines embedding-based retrieval with generative modeling.
- Modular architecture for easy customization.
- Supports various retrieval strategies like similarity search.
- Ideal for use cases like chatbots, document summarization, and question answering.

## Features

- **Text Splitting and Embedding**: Efficiently processes documents using LangChain's `RecursiveCharacterTextSplitter` and `OllamaEmbeddings`.
- **Chroma Database**: Stores embeddings for efficient similarity-based retrieval.
- **Interactive Interface**: Provides a Gradio-powered chat interface for user interaction.
- **Local LLM Support**: Leverages locally hosted Meta Llama 3.2 (3B) model through Ollama integration.
- **Customizable Query Handling**: Processes user queries with context-driven responses.

## Libraries and Requirements

The following dependencies are required for the project:

- `chromadb==0.5.18`
- `langchain-community==0.3.7`
- `langchain-ollama==0.2.0`
- `langchain-chroma==0.1.4`
- `gradio==5.6.0`

Install them using:
```bash
pip install -r requirements.txt
```

## Installation and Usage

### Run Locally

#### Steps to Set Up

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NSU-ECE-ChatBot.git
   cd NSU-ECE-ChatBot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `dataset/cse299data_collect.txt` file is in the correct directory and properly formatted.

4. Start the Ollama server locally:
   ```bash
   ollama start
   ```

5. Run the application:
   ```bash
   python chatbot.py
   ```

6. Access the chat interface at `http://127.0.0.1:7860`.

## Pipeline Workflow

1. **Document Loading**:
   - Loads text data from the specified dataset file.

2. **Text Splitting**:
   - Splits documents into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.

3. **Embedding Creation**:
   - Embeds chunks using `OllamaEmbeddings` to create dense vector representations.

4. **Chroma Database Storage**:
   - Stores embeddings in a Chroma database for similarity-based retrieval.

5. **Retrieval**:
   - Retrieves the most relevant chunks using similarity search.

6. **Generative Response**:
   - Uses the Meta Llama 3.2 model to generate context-informed responses.

7. **Gradio UI**:
   - Provides an interactive interface for querying the chatbot.

## File Structure

- `chatbot.py`: Main script for the chatbot logic.
- `dataset/cse299data_collect.txt`: Data source for the chatbot.
- `requirements.txt`: List of dependencies required to run the project.

## Future Enhancements

- Add support for additional languages and models.
- Expand functionality to answer general university-related queries.
- Integrate cloud-hosted LLMs for broader accessibility.

## Acknowledgments

- Built with the [LangChain](https://www.langchain.com/) framework.
- Powered by [Meta Llama](https://ai.facebook.com/tools/llama/) and [Ollama LLM](https://ollama.com/).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to contribute to this project by submitting issues or pull requests!
