# Open-Reasoning

A local logic engine that implements "System 2" reasoning (Tree-of-Thoughts) using a small open-source model.

The goal was to run complex reasoning chains locally on constrained hardware (**RTX 4050 / 6GB VRAM**) without relying on external APIs.

## How it works
This project wraps a frozen local LLM (Qwen 2.5 7B with 4-bit quantization) in a custom Python control loop. Instead of just generating the next token ("System 1"), the engine forces the model to think hierarchically:

1.  **Branching:** Generates multiple possible logical steps at once.
2.  **Grading:** Evaluates each step using a "Critic" prompt (scoring logic 0.0-1.0).
3.  **Refutation:** If a solution is found, the model enters a verification loop to try and disprove its own answer before accepting it.

## Technical Implementation

### 1. Adaptive Beam Search
I implemented a **Beam Search** algorithm that adapts to the problem difficulty:
* **State Machine:** The search switches between `Exploration` (High Temp), `Verification` (Low Temp), and `Correction` modes based on the current context.
* **Adaptive Scaling:** If the Beam Search fails, it can retry with increased depth and beam width

### 2. Hardware Optimization
* **Engine:** Built on `llama-cpp-python` with CUDA 12.4.
* **VRAM Management:** Running a 7B model on 6GB VRAM is tight. I implemented a strict context window limit (4096 tokens) with a sliding window approach to prevent OOM (Out Of Memory) crashes during deep search trees.
* **Driver Injection:** Includes a custom script to dynamically locate and link NVIDIA drivers (`libcuda.so` / `nvcuda.dll`) at runtime, solving common path issues on Windows dev environments.

## File Structure
* `src/search.py`: The core Beam Search implementation and retry logic.
* `src/chat_engine.py`: Wrapper for the local LLM inference.
* `src/gui.py`: A lightweight visualization tool (CustomTkinter) to watch the reasoning tree grow in real-time.

## Setup & Usage

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/yourusername/open-reasoning.git](https://github.com/yourusername/open-reasoning.git)
    cd open-reasoning
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Model:**
    * Download `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (approx 5GB).
    * Place it in the `models/` folder.

4.  **Run the Visualizer:**
    ```bash
    python src/gui.py
    ```
    * Select **"Flash"** for instant answers (standard LLM behavior).
    * Select "Thinking" for limited depth Beam Search (basic reasoning). 
    * Select "Ultra" for high-depth, high-breadth Beam Search with adaptive retries (full reasoning engine).
## Results
While this experimental engine does not outperform massive commercial reasoning models, 
it significantly outperforms the base model it was built on (Qwen 7B) in tasks requiring multi-step planning.
**Example: Constraint Satisfaction**

> *Riddle: "Arrange the words 'Apple', 'Banana', 'Cherry', 'Date' in a list such that 
'Date' comes before 'Apple', 'Banana' is not last, and 'Cherry' is immediately after 'Date'."*

Flash Mode (Base Model): Fails 100% of the time (hallucinates an order that violates constraints, usually with Banana last).

Ultra Mode: solves the riddle with high success rate by verifying constraints and dividing the problem to steps.

**Hallucination Reduction**
The multi-stage verification loop (Rejection/Correction/Evaluation) means the model is more likely to return "No solution found" rather than hallucinating a confident but incorrect answer, increasing overall reliability.

## Future Work
* Integration of standardized benchmarks (GSM8K, ARC) for quantitative scoring.
* Support for "Best-First Search" to prioritize high-scoring nodes.
* Implementation of MCTS (Monte Carlo Tree Search) to compare against Beam Search.
