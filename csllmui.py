import streamlit as st
from json import load, dump
from openai import OpenAI
from anthropic import Anthropic
from typing import Generator, List

# Initialize OpenAI and Anthropic API clients
openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Initialize Streamlit session state variables
if "chats" not in st.session_state:
    st.session_state.chats = load(open("assets/chats.json", "r"))

if "instructions" not in st.session_state:
    st.session_state.instructions = load(open("assets/instructions.json", "r"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "model" not in st.session_state:
    st.session_state.model = "OpenAI gpt-4o-mini"

if "show_total_tokens" not in st.session_state:
    st.session_state.show_total_tokens = False

def save_chats() -> None:
    """Save the current chats session to a JSON file."""
    dump(st.session_state.chats, open("assets/chats.json", "w"))

def radio_label_formatter(option: int) -> str:
    """Format radio button labels for chat selection.

    Args:
        option (int): Index of the chat in assets/chats.json file.

    Returns:
        str: Name of the chat.
    """
    return st.session_state.chats[option]['name']

def create_chat() -> None:
    """Create a new chat and set it as the current chat."""
    st.session_state.chats.append({
        "name": "Untitled chat",
        "messages": [{"role": "info", "content": f"Model set to: <b>{st.session_state['model'].split()[-1]}</b>"}],
        "model": st.session_state.model
    })
    st.session_state.current_chat = len(st.session_state.chats) - 1
    load_messages()

def delete_chat() -> None:
    """Delete the currently selected chat."""
    del st.session_state.chats[st.session_state.current_chat]
    if len(st.session_state.chats) == 0:
        st.session_state.current_chat = None
    elif st.session_state.current_chat != 0:
        st.session_state.current_chat -= 1
        load_messages()
    else:
        load_messages()

def edit_chat_name() -> None:
    """Edit the name of the currently selected chat."""
    st.session_state.chats[st.session_state.current_chat]['name'] = st.session_state.chat_name
    st.session_state.current_chat = st.session_state.current_chat

def load_messages() -> None:
    """Load messages of the currently selected chat into the session state."""
    st.session_state.messages = st.session_state.chats[st.session_state.current_chat]['messages']
    st.session_state.model = st.session_state.chats[st.session_state.current_chat]['model']

def model_change() -> None:
    """Update the chat and session state to reflect the selected model change."""
    if st.session_state.current_chat is not None:
        if st.session_state.messages[-1]['role'] == 'info':
            st.session_state.messages[-1]['content'] = f"Model set to: <b>{st.session_state['model'].split()[-1]}</b>"
        else:
            st.session_state.messages.append({"role": "info", "content": f"Model set to: <b>{st.session_state['model'].split()[-1]}</b>"})
        st.session_state.chats[st.session_state.current_chat]['model'] = st.session_state.model

def get_total_tokens() -> int:
    """Calculate the total number of tokens in assistant messages.

    Returns:
        int: Total tokens used.
    """
    tokens = 0
    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            tokens += message['tokens']
    return tokens


def get_models() -> List[str]:
    """Get a list of available models from OpenAI and Anthropic.

    Returns:
        List[str]: List of model names.
    """
    openai_models = [f"OpenAI {model}" for model in sorted([
        "gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview",
        "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"
    ], reverse=True)]

    anthropic_models = [f"Anthropic {model}" for model in sorted([
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ], reverse=True)]

    return openai_models + anthropic_models

def pills_label_formatter(label: str) -> str:
    """Format labels for pill-style response type selectors.

    Args:
        label (str): Response type label.

    Returns:
        str: Formatted label.
    """
    match label:
        case "default":
            return f":material/description: {label.capitalize()}"
        case "step-by-step":
            return f":material/handyman: {label.capitalize()}"
        case "results only":
            return f":material/target: {label.capitalize()}"
        case "concise":
            return f":material/content_cut: {label.capitalize()}"

def model_response() -> Generator[str, None, None]:
    """
    Generates responses from the selected AI model based on the session state.

    Yields:
        str: The partial or complete response text from the model.
    """
    def _openai(messages):
        stream = openai.chat.completions.create(
            model=st.session_state["model"].split()[-1],
            messages=[{
                    "role": ('user' if st.session_state.model.split()[-1] in ['o1-preview', 'o1-mini'] else 'system'),
                    "content": st.session_state.instructions['dev'] + "\n\n" + st.session_state.instructions[st.session_state.response_type]
            }] + messages,
            temperature=st.session_state.model_temperature,
            stream=True,
            stream_options={"include_usage": True}
        )
        text = ""
        tokens = 0
        for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None:
                    text += content
                    text = text.replace('\\(', '$').replace('\\)', '$')
                    yield text
            if chunk.usage:
                tokens = chunk.usage.total_tokens
        st.session_state.messages.append({"role": "assistant", "content": text, "tokens": tokens})

    def _anthropic(messages):
        text = ""
        with anthropic.messages.stream(
                model=st.session_state["model"].split()[-1],
                system=st.session_state.instructions['dev']+"\n\n"+st.session_state.instructions[st.session_state.response_type],
                messages=messages,
                temperature=st.session_state.model_temperature,
                max_tokens=8192
        ) as stream:
            for content in stream.text_stream:
                text += content
                text = text.replace('\\(', '$').replace('\\)', '$')
                yield text
            tokens = stream.get_final_message().usage.input_tokens + stream.get_final_message().usage.output_tokens
        st.session_state.messages.append({"role": "assistant", "content": text, "tokens": tokens})

    if st.session_state.model_memory:
        memory = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
            if message["role"] != "info"
        ]
    else:
        memory = [{
            "role": st.session_state.messages[-1]["role"],
            "content": st.session_state.messages[-1]["content"]
        }]

    if "OpenAI" in st.session_state.model:
        model = _openai
    elif "Anthropic" in st.session_state.model:
        model = _anthropic
    else:
        model = lambda: []

    response = ""
    for state in model(memory):
        response = state
        if st.session_state.response_streaming:
            yield state
    return response

def main() -> None:
    """
    Main function to initialize the Streamlit application and set up the user interface.
    """
    st.set_page_config(page_title="CSLLMUI", layout="wide")
    st.write(f"<style>{open('assets/styles.css', 'r').read()}</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.logo(image="assets/csllmui.png", size="large")
        st.button(label="Create new chat", use_container_width=True, on_click=create_chat, icon=":material/add:")

        if st.session_state.current_chat is not None:
            with st.popover(label="Edit chat name", use_container_width=True, icon=":material/edit:"):
                st.text_input(
                    label="Enter new chat name:",
                    value=st.session_state.chats[st.session_state.current_chat]['name'],
                    on_change=edit_chat_name,
                    key="chat_name"
                )
            st.button(label="Delete selected chat", use_container_width=True, on_click=delete_chat, icon=":material/delete:")
            if st.session_state.show_total_tokens:
                st.button(
                    label=f"Total chat tokens: {get_total_tokens()}",
                    use_container_width=True,
                    disabled=True,
                    icon=":material/token:"
                )

        with st.expander(label="Chat history", expanded=(st.session_state.current_chat is None), icon=":material/list:"):
            st.radio(
                label="Select chat:",
                label_visibility="collapsed",
                options=[x for x in range(len(st.session_state.chats))],
                key="current_chat",
                format_func=radio_label_formatter,
                on_change=load_messages
            )

        if st.session_state.current_chat is not None:
            with st.expander(label="Settings", expanded=True, icon=":material/settings:"):
                st.selectbox(
                    label=":material/robot: Model:",
                    options=get_models(),
                    key="model",
                    on_change=model_change,
                    help="Select the AI model to use for generating responses. Options include various models available in the system, such as OpenAI or Anthropic models. You can change models within a chat without losing memory or chat history."
                )
                st.slider(
                    label=":material/thermostat: Model temperature:",
                    min_value=0.0,
                    max_value=2.0,
                    key="model_temperature",
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="Adjust the model's temperature to control response creativity. Lower values (e.g., 0.0) make responses more focused and deterministic, while higher values (e.g., 2.0) increase randomness and creativity. Default is 1.0 for a balanced approach."
                )
                st.pills(
                    label="Response type:",
                    options=["default", "step-by-step", "results only", "concise"],
                    selection_mode="single",
                    default="default",
                    format_func=pills_label_formatter,
                    key="response_type",
                    help="Select how the model structures its responses based on your preferences for this chat."
                )
                st.write("<p style='font-size: 0.9rem; margin-bottom: 0.15rem;'>Others:</style>", unsafe_allow_html=True)
                with st.container(key="OTHERS"):
                    st.toggle(
                        label=":material/memory: Model memory",
                        key="model_memory",
                        value=True,
                        help="Enable or disable the model's memory. When enabled, the model retains context from previous messages within the chat to provide more coherent and context-aware responses."
                    )
                    st.toggle(
                        label=":material/text_fields_alt: Stream responses",
                        key="response_streaming",
                        value=True,
                        help="Enable or disable response streaming. When enabled, the model's responses will be displayed incrementally as they are generated."
                    )
                    st.toggle(
                        label=":material/token: Show chat tokens count",
                        key="show_total_tokens",
                        help="Enable or disable the display of the total token count for the current chat. This helps track token usage during interactions."
                    )

    if st.session_state.current_chat is None:
        st.markdown("<h1>Custom Streamlit Large Language Models <span style='opacity: 10%'>(Power)</span> User Interface</h1>", unsafe_allow_html=True)
        with st.container(key="DESCRIPTION"):
            st.markdown(open('assets/description.md', 'r').read())
            st.write("<h5></h5>", unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            if message['role'] == 'info':
                st.caption(f"<p style='text-align: right; font-size: 1rem;'>{message['content']}</p>", unsafe_allow_html=True)
                st.write("")
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input():
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                with st.empty():
                    for response in model_response():
                        st.markdown(response)

    save_chats()

if __name__ == '__main__':
    main()