# Open-Reasoning

A local logic engine that implements "System 2" reasoning (Tree-of-Thoughts) using a small open-source model.

The goal was to run complex reasoning chains locally on constrained hardware (RTX 4050 / 6GB VRAM) without relying on external APIs.

## How it works
This project wraps a frozen local LLM (Qwen 2.5 7B with 4 bit quantization) in a python control loop. Instead of just generating the next token, the code:
1.  Generates multiple possible next steps.
2.  Evaluates them using a scoring prompt.
3.  Selects the best path (beam Search/MCTS).

## Technical Implementation
1.  Engine: Built on `llama-cpp-python` with CUDA 12.4.
2.  Hardware Optimization: model running on GPU with limited VRAM, handled by using limited context window (4096 tokens) with a sliding window approach to prevent crashes.
3.  Driver Injection: Includes a custom script to dynamically locate and link NVIDIA drivers on Windows systems 
where PATH variables are often broken.

## Setup
1.  Clone the repo.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the model (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`) and put it in a `models/` folder.
4.  Run the chat loop:
    ```bash
    python src/chat_engine.py
    ```