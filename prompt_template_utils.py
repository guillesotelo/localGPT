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
Answer only based on the provided context. If the answer is not in the context, do not answer and state that you cannot find the reference. 
Respond concisely and directly, avoiding elaboration, unnecessary details, formalities, or greetings. 
Prioritize code responses in C++, then C, then Python, unless another language is requested or appears in the context. 
Return acronyms as-is if their meaning is unclear from the context.
"""

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


def get_chat_prompt(prompt):
    return f"""
            {system_prompt}
            {prompt}
            """


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False, use_context=True, user_prompt=''):
    if promptTemplate_type == "llama":
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

    elif promptTemplate_type == "llama3":

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

    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "[INST] ", " [/INST]"
        BOG, EOG = "<s> ", " </s>"
        if history:
            prompt_template = (
                B_INST
                + BOG
                + system_prompt
                + EOG
                + """
    
            Context: {history} \n {context}
            User: {input}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "input"], template=prompt_template)
        else:
            if use_context:    
                prompt_template = (
                    B_INST
                    + BOG
                    + system_prompt
                    + EOG
                    + """
                
                Context: {context}
                User: {input}"""
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
            else:
                prompt_template = (
                    B_INST
                    + BOG
                    + system_prompt
                    + EOG
                    + """
                
                User: {input}"""
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
                # prompt = get_chat_prompt(user_prompt)
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

    # print('\n')
    # print(f"\nPrompt template: {prompt}\n\n")
    # print('\n')

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