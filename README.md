# CSLLMUI - Custom Streamlit Large Language Models User Interface

CSLLMUI is a user-friendly interface for interacting with multiple AI models, such as OpenAI and Anthropic. It supports chat-based interactions, model switching, token tracking, and response customization.

## Features

- **Multi-Model Support**: Seamlessly switch between OpenAI and Anthropic models within one chat.
- **Persistent Chat History**: Save and reload chat sessions for continued conversations.
- **Response Customization**: Adjust response types, model temperature, and memory settings to suit your needs.
- **Token Tracking**: Monitor total token usage for each chat session.
- **Interactive Interface**: Create, rename, or delete chats, and configure settings directly from the sidebar.

## Customization

- **Instructions**: Modify the `assets/instructions.json` file to customize predefined model behaviors.
- **Styling**: Edit the `assets/styles.css` file for minor UI customizations.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/miuoshdot/csllmui.git
    cd csllmui
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up API keys in a `.streamlit/secrets.toml` file:
    ```toml
    OPENAI_API_KEY = "your-openai-api-key"
    ANTHROPIC_API_KEY = "your-anthropic-api-key"
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## License

This project is licensed under the MIT License.
