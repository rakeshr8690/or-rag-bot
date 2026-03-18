"""
LLM handler for generating responses using either:
 - FREE local HuggingFace models (transformers), or
 - Hugging Face Inference API (hosted inference)
"""

from typing import List, Dict, Optional
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from ..config.settings import LLM_MODEL, USE_LOCAL_LLM, HF_API_TOKEN, HF_MAX_NEW_TOKENS
from ..config.prompts import (
    SYSTEM_PROMPT, CONVERSATION_PROMPT, FORMULATION_PROMPT, SENSITIVITY_ANALYSIS_PROMPT,
    CODE_GENERATION_PROMPT, EXPLANATION_PROMPT, MERGED_FORMULATION_PROMPT, FORMULATION_EDIT_PROMPT
)
from ..utils.formatter import ResponseFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMHandler:
    """Handle LLM interactions via local model or HF Inference API."""
    
    def __init__(self, model_name: str = LLM_MODEL, use_local: bool = USE_LOCAL_LLM):
        """
        Initialize FREE local LLM.
        
        Args:
            model_name: HuggingFace model name (e.g., mistralai/Mistral-7B-Instruct-v0.2)
        """
        self.model_name = model_name
        self.use_local = use_local
        self.last_formulation_tex: str = ""
        self.formatter = ResponseFormatter()
        if self.use_local:
            self._init_local_model()
            
    def _init_local_model(self):
        """Initialize FREE local model using transformers with quantization."""
        try:
            logger.info(f"Loading FREE local model: {self.model_name}")
            logger.info("This will download the model on first run (~7-14GB)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            has_gpu = torch.cuda.is_available()
            
            if has_gpu:
                model_name_lower = self.model_name.lower()
                is_small_model = any(keyword in model_name_lower for keyword in [
                    "tinyllama", "1.1b", "1.5b", "phi-2", "qwen2.5-1.5b"
                ])

                if is_small_model:
                    logger.info("GPU detected! Loading small model in float16 (no 4-bit quantization)")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                else:
                    logger.info("GPU detected! Using 4-bit quantization for speed")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
            else:
                logger.info("No GPU detected. Loading on CPU (slower)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            logger.info("✓ FREE local model loaded successfully!")
            logger.info(f"Model size: ~{self._get_model_size()}")
            
        except Exception as e:
            logger.error(f"Failed to load FREE model: {e}")
            logger.error("Try installing: pip install accelerate bitsandbytes")
            raise
    
    def _get_model_size(self):
        """Estimate model size in memory."""
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            size_gb = (param_count * 4) / (1024**3)
            return f"{size_gb:.1f}GB"
        except:
            return "Unknown"
            
    def generate_response(self, user_query: str, context: str, 
                         prompt_template: str = CONVERSATION_PROMPT,
                         max_tokens: int = HF_MAX_NEW_TOKENS) -> str:
        """
        Generate response using FREE local model.
        
        Args:
            user_query: User's question
            context: Retrieved context from vector store
            prompt_template: Template for the prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response
        """
        formatted_prompt = (
            prompt_template
            .replace('{retrieved_context}', context)
            .replace('{user_query}', user_query)
        )
        
        if self.use_local:
            return self._generate_local(formatted_prompt, max_tokens)
        else:
            return self._generate_hf_inference(formatted_prompt, max_tokens)
            
    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        """Generate response using FREE local model."""
        try:
            if "mistral" in self.model_name.lower():
                full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]"
            elif "llama" in self.model_name.lower():
                full_prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"
            elif "tinyllama" in self.model_name.lower():
                full_prompt = f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
            else:
                full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            
            try:
                model_max = int(getattr(self.tokenizer, "model_max_length", 1024))
            except Exception:
                model_max = 1024
            max_input_len = min(max(1024, model_max), 16384)
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len
            )
            
            is_4bit = bool(getattr(self.model, "is_loaded_in_4bit", False))
            is_8bit = bool(getattr(self.model, "is_loaded_in_8bit", False))
            if not (is_4bit or is_8bit):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            logger.info("Generating response with FREE model (this may take 5-20 seconds)...")
            
            with torch.no_grad():
                small_model = any(k in self.model_name.lower() for k in ["tinyllama", "1.1b", "1.5b", "phi-2"]) 
                HARD_CAP = min(max(1024, HF_MAX_NEW_TOKENS), 32000)
                if small_model:
                    recommended_cap = 512
                else:
                    recommended_cap = HARD_CAP

                new_tokens = int(min(max_tokens, recommended_cap))
                logger.info(f"Requested max_tokens={max_tokens}, using max_new_tokens={new_tokens}")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            cleaned = response
            if "[/INST]" in response:
                cleaned = response.split("[/INST]")[-1].strip()
            elif "<|assistant|>" in response:
                cleaned = response.split("<|assistant|>")[-1].strip()
            elif full_prompt in response:
                cleaned = response.replace(full_prompt, "").strip()
            response = cleaned if cleaned else response.strip()
            
            logger.info("✓ Response generated successfully!")
            return response
            
        except Exception as e:
            logger.error(f"Local model error: {e}")
            raise

    def _generate_hf_inference(self, prompt: str, max_tokens: int) -> str:
        """Generate response using Hugging Face Inference API.

        Prefers huggingface_hub.InferenceClient (chat.completions) when available,
        otherwise falls back to direct HTTPS calls to the router endpoint.
        """
        try:
            if not HF_API_TOKEN:
                raise ValueError("HF_API_TOKEN not set in environment")
            try:
                from huggingface_hub import InferenceClient  # type: ignore
                client = InferenceClient(api_key=HF_API_TOKEN)
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=min(max_tokens, HF_MAX_NEW_TOKENS),
                    temperature=0.7,
                    top_p=0.9,
                )
                return completion.choices[0].message.content.strip()  # type: ignore[attr-defined]
            except Exception:
                import requests, time
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": min(max_tokens, HF_MAX_NEW_TOKENS),
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "return_full_text": False,
                    "wait_for_model": True
                }
            }
            def call_model(model_name: str):
                url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
                for attempt in range(3):
                    r = requests.post(url, headers=headers, json=payload, timeout=120)
                    if r.status_code == 202:
                        time.sleep(2 + attempt)
                        continue
                    if r.status_code == 404:
                        return None
                    r.raise_for_status()
                    return r
                return None

            resp = call_model(self.model_name)
            if resp is None:
                for fallback in [
                    "google/gemma-2-2b-it",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "Qwen/Qwen2.5-1.5B-Instruct"
                ]:
                    logger.warning(f"HF Inference 404 for {self.model_name}. Trying fallback: {fallback}")
                    resp = call_model(fallback)
                    if resp is not None:
                        self.model_name = fallback
                        break
            if resp is None:
                raise RuntimeError("No available HF Inference model (404 on all candidates)")
            data = resp.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            return str(data)
        except Exception as e:
            logger.error(f"HF Inference API error: {e}")
            raise
            
    def formulate_problem(self, user_query: str, context: str) -> str:
        """Merged formulation + sensitivity-style reasoning in one answer."""
        response = self.generate_response(
            user_query, context, MERGED_FORMULATION_PROMPT, max_tokens=800
        )
        
        response = self.formatter.format_response(response)
        
        is_valid, warnings = self.formatter.validate_formulation(response)
        if warnings:
            logger.warning(f"Formulation validation warnings: {'; '.join(warnings)}")
        
        try:
            import re
            m = re.search(r"\$\$[\s\S]*?\$\$", response)
            if m:
                self.last_formulation_tex = m.group(0)
        except Exception:
            pass
        return response
        
    def analyze_sensitivity(self, problem_context: str, 
                           sensitivity_query: str, context: str) -> str:
        """Perform a minimal-edit update to the previous formulation if available; otherwise fall back."""
        if self.last_formulation_tex:
            prompt = FORMULATION_EDIT_PROMPT.replace('{previous_formulation}', self.last_formulation_tex).replace('{modification}', sensitivity_query)
            updated = self.generate_response(sensitivity_query, context, prompt_template=prompt, max_tokens=600)
            try:
                import re
                m = re.search(r"\$\$[\s\S]*?\$\$", updated)
                if m:
                    self.last_formulation_tex = m.group(0)
            except Exception:
                pass
            return updated
        else:
            prompt = SENSITIVITY_ANALYSIS_PROMPT.replace('{problem_context}', problem_context).replace('{sensitivity_query}', sensitivity_query).replace('{retrieved_context}', context)
            return self.generate_response(sensitivity_query, context, prompt_template=prompt, max_tokens=800)
        
    def generate_code(self, problem_description: str, context: str,
                     solver_library: str = "PuLP") -> str:
        """Generate Python code for solving the problem."""
        prompt = CODE_GENERATION_PROMPT.format(
            problem_description=problem_description,
            solver_library=solver_library,
            retrieved_context=context
        )
        
        return self.generate_response(
            problem_description, context,
            prompt_template=prompt, max_tokens=1200
        )
        
    def explain_concept(self, technical_content: str, context: str) -> str:
        """Explain technical content in simple terms."""
        prompt = EXPLANATION_PROMPT.format(
            technical_content=technical_content,
            retrieved_context=context
        )
        
        return self.generate_response(
            technical_content, context,
            prompt_template=prompt, max_tokens=800
        )