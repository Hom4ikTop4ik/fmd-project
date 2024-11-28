import imgparser
import os

# path to the directory with images
imageDir = "./samples/images/"  

# path to the directory with 2d coordinates
coordsDir = "./samples/coords2d/" 

circimgDir = "./samples/circimages/"

circDir = "./samples/circles/"

imageSize = (512, 512)
i = 2178
skipcnt = 300
for condfile in os.listdir(circDir):
    try:
        rawCoords = imgparser.parse(circDir + condfile)
    except:
        print("exception, image " + str(i) + "skipped")
        skipcnt+=1
        i+=1
        continue
    if(len(rawCoords) != 68):
        skipcnt += 1
        istr = str(i).zfill(5)
        skipstr = str(skipcnt).zfill(5)
        print("image " + istr + " skipped (" + skipstr + "), length of coords is " + str(len(rawCoords)))
        i += 1
        continue
    else:
        print(condfile)
    coordFile = open(coordsDir + "condcords" + str(i).zfill(5) + ".txt", "w")
    for coords in rawCoords:
        coordFile.write(str(round(coords[0], 4)) + " " + str(round(coords[1], 4))  + '\n')
    coordFile.close()
    i += 1
    #imgparser.showPointsWithIndexes(circimgDir + datafile, rawCoords, imageSize)

print("skip percentage: ", str(round((skipcnt/i) * 100, 0)) + " %")