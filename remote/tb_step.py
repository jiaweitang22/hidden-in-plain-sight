import glob, time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
f = sorted(glob.glob("/content/out/garden/tb/*"))[-1]
ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
tags = ea.Tags()["scalars"]
ev = ea.Scalars(tags[0])
print("tags", tags[:6]); print("last step", ev[-1].step, "wall", round(time.time() - ev[-1].wall_time), "s ago")
print("rate it/s", round((ev[-1].step - ev[0].step) / max(1, ev[-1].wall_time - ev[0].wall_time), 2))
for t in tags:
    if "num_GS" in t or "loss" in t: print(t, round(ea.Scalars(t)[-1].value, 4))
