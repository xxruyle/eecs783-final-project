from text.util import easyocr_extract_text
from util import tests
import time 
import easyocr 
import warnings

warnings.filterwarnings('ignore')

EASY_OCR_READER = easyocr.Reader(['en'])

def run_easyocr_test():
    compute_times = []
    for img_filepath, expected_output in tests:
        start_time = time.time()

        results = easyocr_extract_text(EASY_OCR_READER=EASY_OCR_READER, img_filepath=img_filepath)

        end_time = time.time()
        compute_time = end_time - start_time 
        compute_times.append(compute_time)

        print(f"{img_filepath}:\n", results, f"\n| took {compute_time} seconds")
    
    avg_time = sum(compute_times)/len(tests)
    print("Average Compute Time (s): ", avg_time)