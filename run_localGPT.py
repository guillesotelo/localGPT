import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from queue import Queue
from langchain.llms import LlamaCpp
from prompt_template_utils import get_prompt_template
from utils import get_embeddings
from dotenv import load_dotenv
load_dotenv()

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MODEL_PATH,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
    N_BATCH,
    CONTEXT_WINDOW_SIZE,
    TEMPERATURE,
    R_PENALTY,
    N_GPU_LAYERS,
    TOP_P,
    TOP_K
)

import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    LOGGING.info(f"Loading Model: {model_id}, on: {device_type}...")
    # LOGGING.info("############ This action can take a few minutes!")

    # Uncomment to download based on constants config
    if model_basename:
        if ".gguf" in model_basename.lower() or ".ggml" in model_basename.lower():
            model_path, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model_path, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model_path, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model_path, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # LOGGING.info(f"############ model_path: {model_path}, tokenizer: {tokenizer}")

    if not model_path:
        raise ValueError("Model path is not set or returned.") 

    try:
        generation_config = GenerationConfig.from_pretrained(model_id)
    except EnvironmentError:
        LOGGING.warning("generation_config.json not found. Using default generation configuration.")
        generation_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=float(os.getenv("TEMPERATURE", 0.2)),
            top_p=float(os.getenv("TOP_P", 0.95)),
            repetition_penalty=1.1,
        )

    # LOGGING.info(f"############ Final model_path: {model_path}")
    
    model_kwargs = {
        'precision': "fp16",
        'tfs_z': 1.0,
        'offload_kqv': True
    } if device_type == 'cuda' else {}

    try:
        llm = LlamaCpp(
            model_path=model_path or MODEL_PATH,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            n_ctx=CONTEXT_WINDOW_SIZE,
            n_batch=N_BATCH,
            callbacks=[StreamingStdOutCallbackHandler()],
            streaming=True,
            repeat_penalty=R_PENALTY,
            n_gpu_layers=N_GPU_LAYERS,
            top_p=TOP_P,
            top_k=TOP_K,
            verbose=True,
            device=device_type,
            model_kwargs=model_kwargs
        )
    except KeyError as e:
        LOGGING.error(f"KeyError in LlamaCpp initialization: {e}")
        raise ValueError(f"Failed to initialize model with model_path: {MODEL_PATH}")

    LOGGING.info(f"*** Local LLM successfully loaded on {device_type} ***")
    
    return llm

# This is the one that works
# def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
#     LOGGING.info(f"############ Loading Model: {model_id}, on: {device_type}")
#     LOGGING.info("############ This action can take a few minutes!")
    
#     # Load model and tokenizer based on model_basename
#     if model_basename:
#         if ".gguf" in model_basename.lower() or ".ggml" in model_basename.lower():
#             model_path, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
#         elif ".awq" in model_basename.lower():
#             model_path, tokenizer = load_quantized_model_awq(model_id, LOGGING)
#         else:
#             model_path, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
#     else:
#         model_path, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)
    
#     LOGGING.info(f"############ model_path: {model_path}, tokenizer: {tokenizer}")
    
#     # Validate that model_path is defined
#     if not model_path:
#         raise ValueError("Model path is not set or returned.")
    
#     # Initialize generation config with default if not found
#     try:
#         generation_config = GenerationConfig.from_pretrained(model_id)
#     except EnvironmentError:
#         LOGGING.warning("generation_config.json not found. Using default generation configuration.")
#         generation_config = GenerationConfig(
#             max_new_tokens=1000,
#             temperature=0.2,
#             top_p=0.95,
#             repetition_penalty=1.1,
#         )
    
#     # Add logging here to check the value of model_path
#     LOGGING.info(f"############ Final model_path: {model_path}")
    
#     # Load LlamaCpp model
#     try:
#         llm = LlamaCpp(
#             model_path='./models/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf',  # Ensure model path is passed
#             max_tokens=generation_config.max_new_tokens,
#             temperature=generation_config.temperature,
#             top_p=generation_config.top_p,
#             repeat_penalty=generation_config.repetition_penalty,
#             device=device_type,
#             n_ctx=2048
#         )
#     except KeyError as e:
#         LOGGING.error(f"############  KeyError in LlamaCpp initialization: {e}")
#         raise ValueError(f"############ Failed to initialize model with model_path: {model_path}")
    
#     LOGGING.info("############ Local LLM Loaded")
#     return llm





# ORIGINAL load_model
# def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
#     """
#     Select a model for text generation using the HuggingFace library.
#     If you are running this for the first time, it will download a model for you.
#     subsequent runs will use the model from the disk.

#     Args:
#         device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
#         model_id (str): Identifier of the model to load from HuggingFace's model hub.
#         model_basename (str, optional): Basename of the model if using quantized models.
#             Defaults to None.

#     Returns:
#         HuggingFacePipeline: A pipeline object for text generation using the loaded model.

#     Raises:
#         ValueError: If an unsupported model or device type is provided.
#     """
#     logging.info(f"Loading Model: {model_id}, on: {device_type}")
#     logging.info("This action can take a few minutes!")
    
#     if model_basename is not None:
#         if ".gguf" in model_basename.lower():
#             llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
#             return llm
#         elif ".ggml" in model_basename.lower():
#             model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
#         elif ".awq" in model_basename.lower():
#             model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
#         else:
#             model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
#     else:
#         model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

#     # Load configuration from the model to avoid warnings
#     generation_config = GenerationConfig.from_pretrained(model_id)
#     # see here for details:
#     # https://huggingface.co/docs/transformers/
#     # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

#     # Create a pipeline for text generation
#     if device_type == "hpu":
#         from gaudi_utils.pipeline import GaudiTextGenerationPipeline

#         pipe = GaudiTextGenerationPipeline(
#             model_name_or_path=model_id,
#             max_new_tokens=1000,
#             temperature=0.2,
#             top_p=0.95,
#             repetition_penalty=1.15,
#             do_sample=True,
#             max_padding_length=5000,
#         )
#         pipe.compile_graph()
#     else:
#         pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_length=MAX_NEW_TOKENS,
#         temperature=0.2,
#         # top_p=0.95,
#         repetition_penalty=1.15,
#         generation_config=generation_config,
#     )

#     local_llm = HuggingFacePipeline(pipeline=pipe)
#     logging.info("Local LLM Loaded")

#     return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.

    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """
    if device_type == "hpu":
        from gaudi_utils.embeddings import load_embeddings

        embeddings = load_embeddings()
    else:
        embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            # callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            # callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="mistral",
    type=click.Choice(
        ["llama3", "llama", "mistral", "non_llama"],
    ),
    help="model type, llama3, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
def main(device_type, show_sources, use_history, model_type, save_qa):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.

    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(query, answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
