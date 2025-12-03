from text.vlm_test import run_vlm_test
from text.easyocr_test import run_easyocr_test
from pin.cv_detect import run_cv_img_detect
from pin.cnn_detect import run_cnn_img_detect

import numpy as np
import matplotlib.pyplot as plt

import matplotlib

def run_text_tests():
  run_easyocr_test()
  run_vlm_test()

def run_img_tests():
  run_cv_img_detect()
  run_cnn_img_detect()

def run_tests():
  run_img_tests()
  run_text_tests()

if __name__ == "__main__":
  #run_tests()
  #run_cnn_img_detect()
  run_cv_img_detect()