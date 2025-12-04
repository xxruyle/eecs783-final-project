from pin.cnn_detect import run_cnn_img_detect
from pin.depth_detect_pins import run_depth_detect

def run_img_tests():
  print
  
  print("="*80 + "\n" + f"Depth Based Check")
  run_depth_detect()
  print("="*80 + "\n" + f"Neural Net based check")
  run_cnn_img_detect()


if __name__ == "__main__":
  run_img_tests()