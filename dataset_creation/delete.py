import os 

#add script to registry/dataset
images_dir = 'images'
coords_dir = 'coords'

image_files = os.listdir(images_dir)

for image_file in image_files:

    if image_file.startswith('dataimg') and image_file.endswith('.jpeg'):

        number = image_file[7:12]  

        coords_file = f'condcords{number}_3d.txt'

        if not os.path.exists(os.path.join(coords_dir, coords_file)):
            os.remove(os.path.join(images_dir, image_file))
