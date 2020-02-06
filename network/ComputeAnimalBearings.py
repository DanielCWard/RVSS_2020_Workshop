import evaluate
from PIL import Image
import json
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import math
import os
class PosedImage:
    def __init__(self, json_line):
        img_dict = json.loads(json_line)
        self.pose = np.array(img_dict["pose"])
        self.img_name = img_dict["imgfname"]


    def write_bearings(self, neuralnet, bearings_file, folder_name=""):
        # Obtain neural net output
        img = Image.open(folder_name+self.img_name)
        heatmap = neuralnet.sliding_window(img)

        #neuralnet.visualise_heatmap(heatmap, img, overlay=True)
        _, counts = np.unique(heatmap, return_counts = 1)
        counts[0] = 0
        print(counts)
        Predictedlabel = np.argmax(counts)

        #Filter only largest label
        heatmapFilter1 = np.where(heatmap!=Predictedlabel, 0, heatmap)
        print(heatmap.shape)

        #Calc all label coordinates
        coorLabel = np.where(heatmapFilter1 == Predictedlabel)
        coorLabel = np.column_stack([coorLabel[0], coorLabel[1]])

        bandwidth = estimate_bandwidth(coorLabel, quantile=.2)
        clustering = MeanShift(bandwidth=bandwidth).fit(coorLabel)
        _, counts = np.unique(clustering.labels_, return_counts = 1)
        PredictedArea = np.argmax(counts)
        PredictedAreaCorr = np.where(clustering.labels_ == PredictedArea)
        PredictedAreaCorr = np.asarray(PredictedAreaCorr)
        print(PredictedAreaCorr)

        heatmapFilter2 = np.zeros(heatmap.shape)
        for i in range(PredictedAreaCorr.shape[1]):
            heatmapFilter2[coorLabel[PredictedAreaCorr[0, i], 0], coorLabel[PredictedAreaCorr[0, i], 1]] = Predictedlabel

        #neuralnet.visualise_heatmap(heatmapFilter2, img, overlay=True)
        finalCoor = np.where(heatmapFilter2 == Predictedlabel)
        averageCoor = np.mean(finalCoor, axis=1)

        heatmapFilter3 = np.zeros(heatmap.shape)
        heatmapFilter3[int(averageCoor[0]), int(averageCoor[1])] = Predictedlabel
        #neuralnet.visualise_heatmap(heatmapFilter3, img, overlay=True)

        #Calculate bearing
        realCoorX = averageCoor[1] * img.size[0] / heatmapFilter3.shape[1]
        delta_x = np.abs(realCoorX - img.size[0]/2)

        bearing = math.atan2(delta_x, fx)
        print(bearing)

        # Compute animal bearings here and save to self.animals.
        # Next, you can use all this information to triangulate the animals!

        bearings = {}
        if Predictedlabel == 1:
            bearings["elephant"] = bearing
        if Predictedlabel == 2:
            bearings["llama"] = bearing
        if Predictedlabel == 3:
            bearings["snake"] = bearing
        # For example, finding the llamas:
        # if np.any(heatmap == 2.0):
        #     llama_coords = np.where(heatmap == 2.0)
        #     average_llama = np.mean(llama_coords, axis=1)
        #     bearings["llama"] = ...
        # Now you need to convert this to a horizontal bearing as an angle.
        # Use the camera matrix for this!

        # There are ways to get much better bearings.
        # Try and think of better solutions than just averaging.

        for animal in bearings:
            bearing_dict = {"pose":self.pose.tolist(),
                            "animal":animal,
                            "bearing":bearings[animal]}
            bearing_line = json.dumps(bearing_dict)
            bearings_file.write(bearing_line+'\n')


if __name__ == "__main__":
    cam_calibration = "../calibration/camera_calibration/intrinsic.txt"
    i=0
    print( os.path.abspath(cam_calibration))
    with open( cam_calibration, 'r') as cam_parameter:
        for ln in cam_parameter:
            num_list = ln.split(',')
            if i==0:
                fx = float(num_list[0])
                cx = float(num_list[2])
            if i==1:
                fy = float(num_list[1])
                cy = float(num_list[2])
            i = i + 1
    # Set up the network
    exp = evaluate.Evaluate()

    # Read in the images
    images_fname = "../system_output/images.txt"
    with open(images_fname, 'r') as images_file:
        posed_images = [PosedImage(line) for line in images_file]

    # Compute bearings and write to file
    bearings_fname = "../system_output/bearings.txt"
    with open(bearings_fname, 'w') as bearings_file:
        for posed_image in posed_images:
            posed_image.write_bearings(exp, bearings_file, "../system_output/")
        bearings_file.close()