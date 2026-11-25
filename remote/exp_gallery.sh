#!/bin/bash
# Reproduce every gallery scene (multi-painting clouds) into /content/out/<scene>.
cd /content/hips
two(){ python -m hips.multi_view --targets targets/art_$1.png targets/art_$2.png --views 0,0 90,0 \
         --n 500000 --steps 1500 --out /content/out/$3 --title "$4" --labels "$5" "$6"; }
python -m hips.multi_view --targets targets/art_great_wave.png targets/art_starry_night.png targets/art_pearl_earring.png \
  --views 0,0 90,0 0,-80 --n 600000 --out /content/out/mv3_art --title "Three Paintings" \
  --labels "The Great Wave" "The Starry Night" "Girl with a Pearl Earring"
python -m hips.multi_view --targets targets/art_mona_lisa.png --views 0,0 --n 400000 --steps 1500 \
  --out /content/out/one_mona --title "Mona Lisa" --labels "Mona Lisa"
two grande_jatte impression_sunrise tv_jatte_sunrise "Sunday and Sunrise" "A Sunday on La Grande Jatte" "Impression, Sunrise"
two the_kiss the_scream tv_kiss_scream "The Kiss and the Scream" "The Kiss" "The Scream"
two birth_of_venus night_watch tv_venus_watch "Venus and the Watch" "The Birth of Venus" "The Night Watch"
echo ALL_DONE
