"""
4.08.25
Idea:

RetinaFace allows for face-detection per frame.
I'm guessing they have some sort of script that already handles whole *.mp4 or other audio-video formats.
I need to see how this output looks for their script.

Assuming we are left with detected face-objects per frame as a result, this is where false-positives appear.
Thankfully, false-positives can be discarded by secondary classifier which will be feed with standardized-aligned RetinaFace
output.
For that I need a dataset which has accurate labels whether a detected objects are really a face or just a toy.
Then, I should try the simplest CNN model trained on this dataset and see how well it deals with it.


1. Ask for RetinaFace implementation that they use. -> They simply use Pupil Cloud Face-Mapper: https://cloud.pupil-labs.com/workspaces/76bcd9b6-7049-41d2-8d23-469281a7ab28/recordings
2. Ask whether there is a dataset with manually labeled objects. -> I don't think in the way it should be done, so we will have to construct our own.
3. Ensure that the set of the toys they use is limited - this would help. -> Yup, there is a limited set.
4. Try to adapt to their format, and develop a simple CNN model to find out whether this approach is even valid. -> Format is below.

"""

"""
4.09.25

Neon Face-Mapper output:
https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/face-mapper/

Now, I need the recordings from the head camera.
Using face_positions.csv, I should be able to extract the frames containing the classified faces.
From there I either give someone the task to create sufficient validated dataset, and try some machine learning on it.
"""
# TODO: 5. Get a complete dataset from single registration and face detection. -> Got from Pupil Cloud Workshop

"""
08.09.25

Got the complete dataset.
Scene camera seems to be sampling at 20Hz - can be get from info.json at gaze_frequency
Based on sections.csv located at recording dir, and face detections .csv at Mapper, detected faces have been extracted.

Now, I need to understand what exactly im trying to do with the CNN
"""
# TODO: 6. Understand exactly what is the desired outcome of this classification.
# TODO: 7. Research what approaches are viable, CNN etc.
# 8. Make sure to properly preprocess the dataset e.g. should we discard images which are too similar? -> Okay, we deduplicate at pre02.
# TODO: 9. Serialize the face extraction algorithm, so that we can create huge dataset based on all recordings.
# TODO: 10. Get the external drive with all the recordings data.

"""
11.09.25

Dataset preparation and image preprocessing.

Serialized the image extraction algorithm, handled by pre00.
Standardized the image size and introduced padding box by pre01.
Subsampled the images using threshold time by pre02.
"""
# 11. Create a script which will allow for fast and efficient labeling of each image. -> Okay, pre03.

"""
12.09.25

Dataset balance and labeling.
Crude CNN model

Simple labeling script at pre03.
Dataset directory creation at pre04.

Model at EyeTrackerCNN.
Training of the model at m00.
Classification at m01.
"""

# 12. After extraction from all recordings, you have to choose a subset of the recordings which will have their frames labeled. -> Already done at pre01.
# TODO: 13. Tweak with the model, dataset and choose the correct size of needed labeled dataset.