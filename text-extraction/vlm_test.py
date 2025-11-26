from util import tests, quen_extract_text
import time 


if __name__ == "__main__":
    compute_times = []
    for img_filepath, expected_output in tests:
        start_time = time.time()

        results = quen_extract_text('qwen3-vl:235b-cloud', img_filepath)
        end_time = time.time()
        compute_time = end_time - start_time 
        compute_times.append(compute_time)
        print(f"{img_filepath}:\n  {results} |\n took ({compute_time}) seconds")

    avg_time = sum(compute_times)/len(tests)

    print("Average Compute Time (s): ", avg_time)

