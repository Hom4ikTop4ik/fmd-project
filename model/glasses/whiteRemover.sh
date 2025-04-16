mkdir -p output

for file in *.{png,jpg,jpeg,webp}; do
  [ -e "$file" ] || continue
  ffmpeg -i "$file" -vf "colorkey=0xfdfdfd:0.3:0.1,format=rgba" -c:v png "output/${file%.*}.png"
done