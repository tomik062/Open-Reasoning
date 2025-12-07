import os
import sys
import time
from email.policy import default
from typing import List, Dict
from llama_cpp import Llama

class ChatEngine:
    def __init__(self, model_path = "../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"):
        self.load_drivers()
        self.context_length=4096
        self.answer_length=500
        print(f"⏳ Initializing model (Context: {self.context_length} tokens)...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # force 100% load on GPU
                n_ctx=self.context_length,  # maximum context memory capacity
                verbose=False  # clean output

            )
            print("model ready on GPU")
        except Exception as e:
            print(f"model loading failed: {e}")
            sys.exit(1)

        #init history with a system prompt
        self.history= [
            {"role": "system",
             "content": ("You are a helpful, logical assistant."
                        "Always answer directly and concisely."
                        "If a question is complex, break it down step-by-step before concluding.")
            }
        ]

    def load_drivers(self):
        #load cuda driver. prefer v12.4 if available.
        paths_to_try = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
        ]
        default_path = os.environ.get('CUDA_PATH')
        if default_path:
            paths_to_try.append(os.path.join(default_path, "bin"))

        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                    os.environ["PATH"] = path + ";" + os.environ["PATH"]
                    return
                except:
                    continue

        print("Warning: Could not find CUDA drivers.")

    def _manage_context(self):
        while True:
            #testing context length to check if answer can certainly fit in
            full_prompt = ""
            for msg in self.history:
                full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            tokens = self.llm.tokenize(full_prompt.encode('utf-8'))
            count = len(tokens)
            if count <= (self.context_length - self.answer_length):
                return #enough space

            else:
                print(f"context full ({count}/{self.context_length}). "
                      "deleting the first request and answer...")
                #keep system prompt(history at index 0), delete first user request and ai response
                self.history.pop(1)
                self.history.pop(1)
    def generate_answer(self,context,temperature=0.2, answer_length=None):
        if answer_length is None:
            answer_length = self.answer_length
        try:
            output = self.llm.create_chat_completion(
                messages=context,
                max_tokens=answer_length,  # Cap response length
                temperature=temperature    #Control model randomness
            )
            return output
        except Exception as e:
            print(f"\nError during generation: {e}")
            return None
    def chat_loop(self):
        print("\n💬 Chat Session Started (Type 'exit' to quit)")
        print("-" * 50)

        while True:
            user_input = input("\n👤 YOU: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("👋 Exiting...")
                break

            # add user input
            self.history.append({"role": "user", "content": user_input})

            # ensure we have space for answer
            self._manage_context()

            print("Thinking...")
            #generate answer
            start = time.time()
            output = self.generate_answer(self.history)
            response_time = time.time() - start
            #print response with metrics
            if output:
                response_text = output['choices'][0]['message']['content']
                token_count = output['usage']['completion_tokens']
                speed = token_count / response_time
                print(f"({response_time:.1f} seconds, {speed:.1f} tokens/s)")
                print(f"AI: {response_text}")

                # add chat history to memory
                self.history.append({"role": "assistant", "content": response_text})

# --- Entry Point ---
if __name__ == "__main__":
    engine = ChatEngine()
    engine.chat_loop()