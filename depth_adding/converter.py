import depthFinder
import imgutils
import coordsParser as parser
from mappings import dlibToMeshMapping as idmap
import mappings
import os

# path to the directory with images
imageDir = "samples/images/"  

# path to the directory with 2d coordinates
coordsDir = "samples/coords2d/" 

# path to the output directory, 3d coordinates will be placed there
outputDir = "samples/coords3d/"

imageSize = (1024, 1024)

for imgfile, coordsfile in zip(os.listdir(imageDir), os.listdir(coordsDir)):
    print(imgfile, coordsfile)
    rawCoords = parser.parse(coordsDir + coordsfile)
    depthlist = depthFinder.findDepth('demoface.obj', rawCoords,
                            mappings.lEyeCornerIdImgDlib,
                            mappings.rEyeCornerIdImgDlib,
                            mappings.upNoseIdImgDlib,
                            idmap, 
                            accuracy=100,debug=False)
    depthlist = imgutils.noiseAdding(depthlist, 0.02)

    # uncomment next line to show depth on image
    imgutils.showPointsWithDepth(imageDir + imgfile, coordsDir + coordsfile, depthlist)
    
    # writing 3d coordinates to the output file
    for cords, depth in zip(rawCoords, depthlist):
        cords = imgutils.normalizeCoords(imageSize, cords)
        cords.append(depth)    
    outFilePath = outputDir + imgfile[:-5] + "txt"
    outFile = open(outFilePath, "w")

    for el in rawCoords:
        outFile.write(' '.join([str(round(i, 4)) for i in el]) + '\n')
    

