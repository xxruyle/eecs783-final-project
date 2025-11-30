from text.util import quen_extract_text
from util import ic_marking_tests
import time 
import ollama

OLLAMA_MODEL = 'qwen3-vl:235b-cloud'

def run_vlm_test():
    compute_times = []
    for img_filepath, expected_output in ic_marking_tests:
        start_time = time.time()

        results = quen_extract_text(OLLAMA_MODEL, img_filepath)
        end_time = time.time()
        compute_time = end_time - start_time 
        compute_times.append(compute_time)
        print(f"{img_filepath}:\n", results, f"\n| took {compute_time} seconds")

    avg_time = sum(compute_times)/len(ic_marking_tests)

    print("Average Compute Time (s): ", avg_time)

