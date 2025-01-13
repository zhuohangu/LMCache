import time

import lmcache_vllm
from lmcache_vllm.blend_adapter import (OfflineKVPreCompute,
                                        combine_input_prompt_chunks)
from lmcache_vllm.vllm import LLM, SamplingParams

if __name__ == "__main__":
    context_files = ["chunk1.txt", "chunk2.txt"]
    chunks = []

    for context_file in context_files:
        with open(context_file, "r") as fin:
            context = fin.read()
        chunks.append(context)

    sys_prompt = "Here's a document from the user: "
    question = "Question: What does this document mainly talks about? Answer: "

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
              gpu_memory_utilization=0.7,
              tensor_parallel_size=2)
    sampling_params_generation = SamplingParams(temperature=0.0,
                                                top_p=0.95,
                                                max_tokens=30)

    print(
        "-------------- Pre-computing KV cache for chunks -------------------")
    offline_precompute = OfflineKVPreCompute(llm)
    for chunk in chunks:
        offline_precompute.precompute_kv(chunk)

    time.sleep(3)
    print("Running the real query here!")

    user_prompt = [sys_prompt, chunks[0], chunks[1], question]
    user_prompt = combine_input_prompt_chunks(user_prompt)
    outputs = llm.generate(user_prompt, sampling_params_generation)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Newly generated text: {generated_text!r}")
        ttft = output.metrics.first_token_time \
            - output.metrics.first_scheduled_time
        print(f"Time to first token: {ttft:.3f} seconds")

    # Graceful exit
    lmcache_vllm.close_lmcache_engine()
