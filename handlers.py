import numpy as np
import fractality_detector
from matplotlib import pyplot as plt

def process_input_file(file):
  im = np.array(plt.imread(file, format='jpeg'))
  D = fractality_detector.self_similariy(im)
  print(D)



if __name__ == "__main__":
  file = "./example_input/example_input.jpg"
  process_input_file(file)