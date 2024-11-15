"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# this is specific to Llama-2.

# system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
# Read the given context before answering questions and think step by step. If you can not answer a user question based on 
# the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

# Prompt for Mistral 7B
system_prompt = """
You are a helpful HP Assistant from Volvo Cars. 
You can only answer questions based on the provided context. 
If the answer is not contained in the context, kindly state that the information is not available in the provided context, and do not speculate or provide any outside knowledge. 
Ensure your response is clear, professional, and based solely on the documents provided. 
Provide the information requested directly.
Do not use roleplay conversation.
Respond concisely and in English only.
Here's the question:
"""

def get_chat_prompt(prompt):
    return f"""
            <s>[INST]
            You are a helpful HP Assistant from Volvo Cars.
            Respond briefly and concisely, providing information that is accurate and reliable. If you are uncertain, do not generate speculative or false information.
            Provide the information requested directly.
            Do not use roleplay conversation.
            Respond concisely and in English only.
            Here's the question:
            {prompt}
            [/INST]
            """


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False, use_context=True, user_prompt=''):
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "llama3":

        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
        B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> ", "<|eot_id|>"
        ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            if use_context:    
                prompt_template = (
                    B_INST
                    + system_prompt
                    + """
                
                Context: {context}
                User: {question}"""
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            else:
                prompt_template = (
                    B_INST
                    + get_chat_prompt(user_prompt)
                    + """
                
                Context: {context}
                User: {question}"""
                    + E_INST
                )
                prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
                # prompt = get_chat_prompt(user_prompt)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    print(f"\n\nHere is the prompt used: {prompt}\n\n")

    return (
        prompt,
        memory,
    )