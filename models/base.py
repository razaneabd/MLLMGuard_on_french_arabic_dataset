import sys
sys.path.append('..')
import os
from tqdm import tqdm
import jsonlines

from utils import RESPONSE_DICT

class Mllm:

    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        pass

    def evaluate(self, prompt, filepath):
        pass

    def batch_evaluate(self, args, data):
      import os
      response_list = []
      # Extract category name from the data_path (e.g., 'position-swapping' from 'data/position-swapping')
      category_from_path = os.path.basename(args.data_path).lower()

      for sample in tqdm(data):
          prompt = sample['prompt']
          lan = sample.get('lan', 'unknown')

          # Check if it's a position-swapping case
          if category_from_path == 'position-swapping' and 'reverse_img_url' in sample:
              image_paths = [sample['img_url'], sample['reverse_img_url']]
          # Handle noise-injection / noise-consistency
          elif category_from_path in ['noise-consistency', 'noise-injection']:
              original_img = sample['img_url']
              base_name = os.path.basename(original_img)
              dir_name = os.path.dirname(original_img)
              name, ext = os.path.splitext(base_name)
              noise_img = os.path.join(dir_name, f"{name}_noise{ext}")
              image_paths = [original_img, noise_img]
          else:
              image_paths = [sample['img_url']]

          for image in image_paths:
              res = RESPONSE_DICT.copy()
              res['prompt'] = prompt
              res['img_url'] = image
              res['lan'] = lan

              try:
                  response = self.evaluate(prompt, image)
                  res['response'] = response
              except Exception as e:
                  print(f'Image {image} Error: {e}')
                  res['response'] = 'Error'

              if args.verbose:
                  print(res)

              response_list.append(res)

      with jsonlines.open(args.save_path, 'w') as writer:
          writer.write_all(response_list)
