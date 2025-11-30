IC_IMAGES_PATH = "./ic-images"


ic_marking_tests = [
    (f'{IC_IMAGES_PATH}/A-J-28SOP-03F-SM.png',  "cy8c27443-24pvxi2001b05cyp603161c"),
    (f'{IC_IMAGES_PATH}/C-T-08DIP-11F-SM.png',  "92aet6g3adc0732ccn"),
    (f'{IC_IMAGES_PATH}/C-T-48QFP-19F-SM.png',  "stm32f103c8t6991rx019umys99008e42"),
    (f'{IC_IMAGES_PATH}/C-T-48QFP-20F-SM.png',  "stm32f103c8t6991uj019umys99009e42")
] # tuple containing image filepath and the visible text on the ic 

DEFECT_IMAGES_PATH = "./defect-images"

defect_images = [
  f'{DEFECT_IMAGES_PATH}/A-D-64QFP-14B-SM.png',
  f'{DEFECT_IMAGES_PATH}/A-D-64QFP-15B-SM.png',
  f'{DEFECT_IMAGES_PATH}/A-J-28SOP-03F-SM.png',
  f'{DEFECT_IMAGES_PATH}/C-T-28SOP-04F-SM.png',
] # list containing filepath of defect images