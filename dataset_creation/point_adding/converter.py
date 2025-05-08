import depthFinder
import imgparser
from PIL import Image
import pandas as pd 
import io


df = pd.read_parquet('train-00000-of-00003.parquet') 
for i in range(0,5):
    img = Image.open(io.BytesIO(df.head()['conditioning_image'][i]['bytes']))
    rawCoords = imgparser.parse(img)
    depthFinder.findDepth('demoface.obj', rawCoords,
                          1107, 13,
                          1967, 52,
                          2104, 5, accuracy=100,debug=True)
    