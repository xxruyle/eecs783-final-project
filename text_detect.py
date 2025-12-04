from text.vlm_test import run_vlm_test
from text.easyocr_test import run_easyocr_test

def run_text_tests():
  print("="*80 + "\n" + f"EASY OCR")
  run_easyocr_test()
  print("="*80 + "\n" + f"OLLAMA")
  run_vlm_test()

if __name__ == "__main__":
  run_text_tests()