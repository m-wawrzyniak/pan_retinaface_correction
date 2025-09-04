"""
Idea:

RetinaFace allows for face-detection per frame.
I'm guessing they have some sort of script that already handles whole *.mp4 or other audio-video formats.
I need to see how this output looks for their script.

Assuming we are left with detected face-objects per frame as a result, this is where false-positives appear.
Thankfully, false-positives can be discarded by secondary classifier which will be feed with standardized-aligned RetinaFace
output.
For that I need a dataset which has accurate labels whether a detected objects are really a face or just a toy.
Then, I should try the simplest CNN model trained on this dataset and see how well it deals with it.


TODO: 1. Ask for RetinaFace implementation that they use.
TODO: 2. Ask whether there is a dataset with manually labeled objects.
TODO: 3. Ensure that the set of the toys they use is limited - this would help.
TODO: 4. Try to adapt to their format, and develop a simple CNN model to find out whether this approach is even valid.

"""