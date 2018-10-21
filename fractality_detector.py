import sys
import numpy as np
import scipy.stats as stats
# box [xmin,ymin,xmax,ymax]

def box_pixel_mass(im_input, box):
  L = len(box)/2
  mins = np.asarray(box[:L], dtype=np.int)
  maxs = np.asarray(box[L:], dtype=np.int)
  boxed_im_input = im_input[mins[0]:maxs[0], mins[1]:maxs[1]]

  img_input_reduced = boxed_im_input[:, :, 0] >= -1.0
  j = 0
  for i in xrange(2, L):
    _img_input_reduced = np.logical_and(boxed_im_input[:, :, j] < maxs[i],
                                        boxed_im_input[:, :, j] > mins[i])
    img_input_reduced = np.logical_and(img_input_reduced, _img_input_reduced)
    j = j + 1
  box_mass = np.count_nonzero(img_input_reduced)

  return box_mass

def meshgrid_iterator2(*arrays):
# 5 dimensions
  i = [0, 0, 0, 0, 0]
  for i[0] in arrays[0]:
    for i[1] in arrays[1]:
      for i[2] in arrays[2]:
        for i[3] in arrays[3]:
          for i[4] in arrays[4]:
            yield i

def sample_box(greater_box, box_sizes):
  L = len(greater_box)/2
  mins = np.asarray(greater_box[:L], dtype=int)
  maxs = np.asarray(greater_box[L:], dtype=int)
  S = [np.arange(mins[i], maxs[i], box_sizes[i]) for i in range(L)]
  St = tuple(map(tuple, S))
  j = 1
  for I in meshgrid_iterator2(*St):
    _I = np.array(I)
    yield np.concatenate([_I, _I + box_sizes])

def num_of_smaller_boxes(greater_box, box_sizes):
  L = len(greater_box)/2
  mins = np.asarray(greater_box[:L], dtype=int)
  maxs = np.asarray(greater_box[L:], dtype=int)
  return np.product([ len(np.arange(mins[i], maxs[i], box_sizes[i])) for i in range(L)])

def box_count(im_input, this_box):
  L = len(this_box)/2
  sizes = this_box[L:] - this_box[:L] + 1
  child_sizes = np.ceil(np.sqrt(1 + sizes))
  # boxes = [sb for sb in sample_box(this_box, child_sizes)]

  num_boxes = num_of_smaller_boxes(this_box, child_sizes)
  # print(num_boxes)

  m_boxes = np.array([box_pixel_mass(im_input, sb) for sb in sample_box(this_box, child_sizes)])
  M_boxes = np.sum(m_boxes)

  Pe = m_boxes / (np.log(np.product(child_sizes)))
  
  SPe = box_pixel_mass(im_input, this_box) / (np.log(np.product(sizes)))
  
  D = (np.log(np.sum(Pe)) - np.log(SPe) ) / np.log(float(num_boxes))
  
  # no distortion to probabilities

  return D

# self similary by box counting
def self_similariy(im_input, depth=8):
  L = np.shape(im_input)
  H, W, C = L[:]
  box = np.concatenate([[0]* (C+2), L[0:2], [255] * C])
  s_s_d = box_count(np.array(im_input), box)

  return s_s_d

def chaotic_measure(im_input):
  D = self_similariy(im_input)
  return max(abs(np.ceil(D) - D), D - np.floor(D))

if __name__ == "__main__":
  if len(sys.argv) == 2:
    INPUT_FILE =  sys.argv[1]
    # How chaotic!!! :P
    print(chaotic_measure(INPUT_FILE))