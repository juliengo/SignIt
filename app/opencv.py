from run import *

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

imageFile = dir_path+"/test.png"


results = predict_local_image(imageFile)

print(results['predictedTagName'])

