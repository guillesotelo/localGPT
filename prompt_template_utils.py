"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

CHAT_PRESENTATION = os.getenv('CHAT_PRESENTATION','')
# this is specific to Llama-2.

# system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
# Read the given context before answering questions and think step by step. If you can not answer a user question based on 
# the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

# Prompt for Mistral 7B
system_prompt = CHAT_PRESENTATION + """
You must follow these rules:

- Only answer questions using the provided context. If the answer cannot be found clearly and explicitly in the context, respond with: "This question is outside the scope of our documentation."
- Do not guess, infer, or make assumptions based on loosely related information.
- Keep responses concise and direct. Do not include greetings, formalities, or unnecessary elaboration.
- Prefer code responses in this order: C++, then C, then Python, unless another language is explicitly mentioned or appears in the context.
- If an acronym is used and its meaning is not defined in the context, return it as-is without interpretation.
"""

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


def get_prompt_template(system_prompt=system_prompt, model_name=None, history=False, use_context=True, user_prompt=''):
    if model_name == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {input}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "input"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {input}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)

    elif model_name == "llama3":

        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
        B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> ", "<|eot_id|>"
        ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {input}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["history", "context", "input"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {input}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)

    elif model_name == "mistral":
        B_INST, E_INST = "[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + "\n\nHistory: {history} Context: {context}\nUser: {input} "
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "input"], template=prompt_template)
        else:
            if use_context:    
                prompt_template = (
                    B_INST
                    + system_prompt
                    + "\n\nContext: {context}\nUser: {input} "
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
            else:
                prompt_template = (
                    B_INST
                    + system_prompt
                    + "User: {input} "
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
                
    elif model_name == "gpt-oss":
        if history:
            prompt_template = (
                system_prompt
                + "\n\nContext: {history} \n {context}\nUser: {input}\nAnswer:"
            )
            prompt = PromptTemplate(input_variables=["history", "context", "input"], template=prompt_template)
        else:
            if use_context:
                prompt_template = (
                    system_prompt
                    + "\n\nContext: {context}\nUser: {input}\nAnswer (only output the final answer, nothing else):"
                )
                prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
            else:
                prompt_template = (
                    system_prompt
                    + "\n\nUser: {input}\nAnswer:"
                )
                prompt = PromptTemplate(input_variables=["input"], template=prompt_template)

    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {input}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "input"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {input}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="input", memory_key="history")

    print(f"\nPrompt template model: {model_name}\n\n")

    return (
        prompt,
        memory,
    )

def get_sources_template():
    B_INST, E_INST = "[INST] ", " [/INST]"
    prompt_template = (
        B_INST
        + system_prompt
        + """
    
    Context: {context}
    User: {question}"""
        + E_INST
    )
    return PromptTemplate(input_variables=["context", "question"], template=prompt_template)