import depthFinder
import imgparser
import coordsParser as parser
from mappings import dlibToMeshMapping as idmap
import mappings
from PIL import Image
import pandas as pd 
import io

df = pd.read_parquet('train-00000-of-00003.parquet') 
for i in range(2,3):
    #img = Image.open(io.BytesIO(df.head()['conditioning_image'][i]['bytes']))
    rawCoords = parser.parse("a_img5002.txt")
    depthlist = depthFinder.findDepth('demoface.obj', rawCoords,
                          mappings.lEyeCornerIdImgDlib,
                          mappings.rEyeCornerIdImgDlib,
                          mappings.upNoseIdImgDlib,
                          idmap, 
                          accuracy=100,debug=False)
    print(depthlist)
