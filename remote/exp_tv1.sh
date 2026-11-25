#!/bin/bash
# Two-view anamorphosis: one cloud, two pictures, two secret spots 90 degrees apart.
cd /content/hips
python -m hips.two_view --a targets/pair_cs.png --b targets/pair_180.png --out /content/out/tv_cs180 --title "CS / 180" --label_a CS --label_b 180
python -m hips.two_view --a targets/pair_heart.png --b targets/pair_star.png --out /content/out/tv_heart_star --title "Heart / Star" --label_a Heart --label_b Star
python -m hips.two_view --a targets/pair_look.png --b targets/pair_again.png --out /content/out/tv_look_again --title "Look again" --label_a LOOK --label_b AGAIN
python -m hips.two_view --a targets/eye.png --b targets/smiley.png --out /content/out/tv_eye_smiley --title "Eye / Smile" --label_a Eye --label_b Smile
echo ALL_DONE
