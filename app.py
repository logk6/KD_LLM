# app.py
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig
)
import torch
import threading
import time
import os

# ----------------------------
# USER CONFIG - sửa đường dẫn này
# ----------------------------
CHECKPOINT = "TinyLLM-OBQA"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
REPETITION_PENALTY = 1.15

# ----------------------------
# DEVICE
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load tokenizer + detect model type + load appropriate model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(checkpoint):
    # tokenizer (use_fast=False for sentencepiece/T5)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    # detect via config
    config = AutoConfig.from_pretrained(checkpoint)

    is_seq2seq = getattr(config, "is_encoder_decoder", False)

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        model_type = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model_type = "causal"

    # move to device and eval
    model.to(DEVICE)
    model.eval()

    return tokenizer, model, model_type

# Try load
try:
    tokenizer, model, MODEL_TYPE = load_model_and_tokenizer(CHECKPOINT)
except Exception as e:
    st.error(f"Failed to load model/tokenizer from {CHECKPOINT}: {e}")
    raise e

# ----------------------------
# Helper: format prompt from history
# ----------------------------
def build_prompt_from_history(history, model_type):
    """
    history: list of dicts: {"role":"user"|"assistant", "content": "..."}
    For seq2seq (T5), we build a simple conversational prompt.
    For causal, we build a concatenated chat-like prompt.
    """
    if model_type == "seq2seq":
        # simple alternating prefix
        parts = []
        for m in history:
            if m["role"] == "user":
                parts.append("User: " + m["content"])
            else:
                parts.append("Assistant: " + m["content"])
        parts.append("Assistant:")
        prompt = "\n".join(parts)
    else:  # causal
        parts = []
        for m in history:
            role = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        parts.append(
            "Assistant: Please answer in the following format:\n"
            "Correct Answer: <option>\n"
            "Reasoning: <explanation>"
        )
        prompt = "\n".join(parts)
    return prompt

# ----------------------------
# Streaming generator using TextIteratorStreamer
# ----------------------------
def stream_generate(prompt):
    try:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
    except Exception:
        streamer = None

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # gen_kwargs = {
    #     "input_ids": input_ids,
    #     "max_new_tokens": MAX_NEW_TOKENS,
    #     "temperature": TEMPERATURE,
    #     # "top_p": TOP_P,
    #     "do_sample": DO_SAMPLE,
    #     # "repetition_penalty": REPETITION_PENALTY,
    #     # "eos_token_id": tokenizer.eos_token_id,
    #     # "decoder_start_token_id": model.config.decoder_start_token_id,
    #     # "no_repeat_ngram_size": 3,
    # }

    generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.7)
    output_seq = model.generate(inputs, generation_config=generation_config)
    output = tokenizer.decode(output_seq[0], skip_special_tokens=True)
    return output

    # Streaming
    # if streamer is not None:
    #     gen_kwargs["streamer"] = streamer
    #     thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    #     thread.start()
    #     for chunk in streamer:
    #         yield chunk
    #     thread.join()
    # else:
    #     # Non-streaming
    #     outputs = model.generate(**gen_kwargs)
    #     yield tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# Streamlit UI
# ----------------------------
def format_mcq(s: str):
    # ví dụ s là "Question: ...\nOptions:\nA. ...\nB. ..."
    # ta biến thành markdown hợp lệ
    parts = s.split("\n")
    out = []
    for line in parts:
        if line.strip().startswith(("A.", "B.", "C.", "D.")):
            out.append("- " + line.strip())   # bullet list
        else:
            out.append(line)
    return "\n\n".join(out)

st.set_page_config(page_title="Chatbot for Multiple-choice QA", layout="centered", page_icon="💬",)
st.title("💬 Chatbot for Multiple-choice QA")

# Lưu lịch sử
if "messages" not in st.session_state:
    st.session_state.messages = []


# Hiển thị lịch sử hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(format_mcq(msg["content"]))

# Input người dùng
if prompt := st.chat_input("Nhập tin nhắn..."):
    # Hiển thị tin người dùng
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Khung assistant message
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ... 🤔"):
            answer = stream_generate(prompt)
            st.markdown(answer.replace("\n", "<br>"), unsafe_allow_html=True)

    # Lưu phản hồi
    st.session_state.messages.append({"role": "assistant", "content": answer})


# small CSS for scrollbar auto-scroll (best-effort)
st.markdown(
    """
<style>
/* Make the main block scrollable if long */
div.block-container{height: 75vh; overflow:auto;}
</style>
""",
    unsafe_allow_html=True,
)
