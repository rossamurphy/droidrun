import os
import torch
import importlib
import logging
from typing import Any

# Set before importing transformers
# Some potentially problematic optimizations you can try and disable if you hit issues

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# torch._dynamo.config.suppress_errors = True
# torch._inductor.config.triton.cudagraphs = False

# getting weird threading issues using the 3090 ... try this to avoid (Gemini recommended it)
# torch._dynamo.disable()

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from llama_index.core.llms.llm import LLM

# Configure logging
logger = logging.getLogger("droidrun")

# 3090 can do this, so consider setting it
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True


logging.basicConfig(level=logging.INFO)  # Set to INFO to see loading status


def load_llm(provider_name: str, **kwargs: Any) -> LLM:
    """
    Dynamically loads and initializes a LlamaIndex LLM.

    Imports `llama_index.llms.<provider_name_lower>`, finds the class named
    `provider_name` within that module, verifies it's an LLM subclass,
    and initializes it with kwargs.

    Args:
        provider_name: The case-sensitive name of the provider and the class
                       (e.g., "OpenAI", "Ollama", "HuggingFaceLLM").
        **kwargs: Keyword arguments for the LLM class constructor.

    Returns:
        An initialized LLM instance.

    Raises:
        ModuleNotFoundError: If the provider's module cannot be found.
        AttributeError: If the class `provider_name` is not found in the module.
        TypeError: If the found class is not a subclass of LLM or if kwargs are invalid.
        RuntimeError: For other initialization errors.
    """
    if not provider_name:
        raise ValueError("provider_name cannot be empty.")



    if provider_name == "HuggingFaceLLM":
        logger.info("Handling special case for HuggingFaceLLM provider.")
        from llama_index.llms.huggingface import HuggingFaceLLM

        model_name = kwargs.get("model")
        if not model_name:
            raise ValueError("HuggingFaceLLM requires a 'model' argument.")

        if torch.cuda.is_available():
            print("CUDA is available, setting default device map to cuda:0")
            device_map = "cuda:0"
        elif torch.backends.mps.is_available():
            print("MPS is available, setting default device map to mps")
            device_map = "mps"
        else:
            print("No accelerator available, setting default device map to cpu")
            device_map = "cpu"

        model_kwargs = {
            "torch_dtype": kwargs.get("torch_dtype"),
            "device_map": kwargs.get("device_map", device_map),
            "max_memory": kwargs.get("max_memory"),
            "attn_implementation": kwargs.get("attn_implementation") 
        }

        print("Using device map:", model_kwargs["device_map"])
        print("If this differs from the default device map, it has been overridden by the user.")

        if kwargs.get("load_in_4bit"):
            if device_map == "mps":
                raise ValueError("4-bit quantization is not supported on MPS devices.")
            logger.info("4-bit quantization enabled.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        # Filter out any None values before passing to the function
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # 2. Load the model from Hugging Face with the specific arguments
        logger.info(f"Loading model '{model_name}' with args: {model_kwargs.keys()}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # 2.5 compile the model?
        # logger.info("Compiling model for faster inference...")
        # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        # 3. Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 4. Define terminators
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        # 5. Initialize the LlamaIndex wrapper, passing the stopping tokens
        logger.info("Initializing HuggingFaceLLM wrapper with stopping tokens.")
        llm_instance = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            stopping_ids=terminators,  # knowing when to stop
            max_new_tokens=kwargs.get("max_new_tokens", 512),
        )

        return llm_instance


    if provider_name == "OpenAILike":
        module_provider_part = "openai_like"
        kwargs.setdefault("is_chat_model", True)
    if provider_name == "GoogleGenAI":
        module_provider_part = "google_genai"
    else:
        # Use lowercase for module path, handle hyphens for package name suggestion
        lower_provider_name = provider_name.lower()
        # Special case common variations like HuggingFaceLLM -> huggingface module
        if lower_provider_name.endswith("llm"):
            module_provider_part = lower_provider_name[:-3].replace("-", "_")
        else:
            module_provider_part = lower_provider_name.replace("-", "_")
    module_path = f"llama_index.llms.{module_provider_part}"
    install_package_name = f"llama-index-llms-{module_provider_part.replace('_', '-')}"

    try:
        llm_module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        logger.error(f"Module '{module_path}' not found. Try: pip install {install_package_name}")
        raise

    try:
        llm_class = getattr(llm_module, provider_name)
        if not issubclass(llm_class, LLM):
            raise TypeError(f"Class '{provider_name}' is not a valid LLM subclass.")

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        logger.info(f"Initializing {llm_class.__name__} with args: {list(filtered_kwargs.keys())}")
        llm_instance = llm_class(**filtered_kwargs)
        logger.info(f"Successfully loaded and initialized LLM: {provider_name}")
        return llm_instance
    except Exception as e:
        logger.error(f"Failed to initialize {provider_name}: {e}")
        raise


# --- Example Usage ---
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv("/Users/ross/GitHub/private-android-world/.env")
    print("\n--- Loading Anthropic ---")
    try:
        anthropic_llm = load_llm(
            "Anthropic",
            model="claude-3-7-sonnet-latest",
        )
        print(f"Loaded LLM: {type(anthropic_llm)}")
        print(f"Model: {anthropic_llm.metadata}")
    except Exception as e:
        print(f"Failed to load Anthropic: {e}")

    print("\n--- Loading DeepSeek ---")
    try:
        deepseek_llm = load_llm(
            "DeepSeek",
            model="deepseek-reasoner",
            api_key="your api",  # or set DEEPSEEK_API_KEY
        )
        print(f"Loaded LLM: {type(deepseek_llm)}")
        print(f"Model: {deepseek_llm.metadata}")
    except Exception as e:
        print(f"Failed to load DeepSeek: {e}")

    # Example 3: Load Gemini (requires GOOGLE_APPLICATION_CREDENTIALS or kwarg)
    print("\n--- Loading Gemini ---")
    try:
        gemini_llm = load_llm(
            "Gemini",
            model="gemini-2.0-fash",
        )
        print(f"Loaded LLM: {type(gemini_llm)}")
        print(f"Model: {gemini_llm.metadata}")
    except Exception as e:
        print(f"Failed to load Gemini: {e}")

    print("\n--- Loading Gemma 3n with HuggingFaceLLM on Linux ---")
    try:
        if torch.cuda.is_available():
            gemma_llm = load_llm(
                "HuggingFaceLLM",
                model="google/gemma-3n-e4b-it",
                device_map="cuda:0",
                # device_map="mps",
                load_in_4bit=False,
                max_new_tokens=512,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa" # trying this out
            )
            print(f"Successfully loaded LLM: {type(gemma_llm)} on CUDA device.")
        elif torch.backends.mps.is_available():
            gemma_llm = load_llm(
                "HuggingFaceLLM",
                model="google/gemma-3n-e4b-it",
                device_map="mps",
                load_in_4bit=False,
                max_new_tokens=512,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa" # trying this out
            )
            print(f"Successfully loaded LLM: {type(gemma_llm)} on MPS device.")
        else:
            gemma_llm = load_llm(
                "HuggingFaceLLM",
                model="google/gemma-3n-e4b-it",
                device_map="cpu",
                load_in_4bit=True,
                max_new_tokens=512,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa" # trying this out
            )
            print(f"Successfully loaded LLM: {type(gemma_llm)} on CPU device.")

        # 2. Define your question
        question = ("who would win in a fight, a bear with a massive sword, or, a cow with a massive gun?")
        print(f"\nQuestion: {question}")

        # 3. Get the streaming response using the .stream_complete() method
        print("\nResponse:")
        response = gemma_llm.complete(question)
        print(response.text)

    except Exception as e:
            print(f"\nAn error occurred during setup or query: {e}")
