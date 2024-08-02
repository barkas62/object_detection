#
# object_detection_video_stream.py: Simple python script to run AI inference on a video stream.
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# This script runs AI inference on a video source and displays the video stream with annotated results.
# The script take a config.yaml file as an input. Tha config file specifies the following parameters.
# hw_location: where you want to run inference
#     "@cloud" to use DeGirum cloud
#     "@local" to run on local machine
#     IP address for AI server inference
# model_zoo_url: url/path for model zoo
#     cloud_zoo_url: valid for @cloud, @local, and ai server inference options
#     '': ai server serving models from local folder
#     path to json file: single model zoo in case of @local inference
# model_name: name of the model for running AI inference
# video_source: video source for inference
#     camera index for local camera
#     URL of RTSP stream
#     URL of YouTube Video
#     path to video file (mp4 etc)

import yaml
import time

import degirum as dg
import degirum_tools


class Detector:
    # how much time (in sec) should elapse with nothing detected, so that de decide that there is nothing indeed
    _absent_time_th = 2

    def __init__(self, labels: set, score_th: float, time_th: float):
        """
        :param labels: set of labels to be detected
        :param score_th: score threshold
        :param time_th: time threshold
        """
        self.labels: set = labels
        self.score_th: float = score_th
        self.time_th : float = time_th

        self.time_detected: float = -1.0   # time of first detection
        self.detected: bool = False        # is something detected
        self.detections = list()           # list of last verified detections

    def get_detections(self, inference_result: dict):
        """
        :param inference_result: model output
        :return: list of verified detections, each having the form {'label':label, 'bbox': bbox}
        """
        curr_detections = list()  # list of verified detections

        curr_time: float = time.time()
        is_something_detected: bool = False
        for result in inference_result.results:
            bbox = result['bbox']
            label = result['label']
            score = result['score']
            if label in self.labels and score > self.score_th:
                is_something_detected = True
                if self.time_detected > 0.0 and curr_time - self.time_detected > self.time_th:
                    #print(f"verified detection at {curr_time}")
                    curr_detections.append({'label': label, 'bbox': bbox})

        if self.time_detected > 0.0 and not is_something_detected:
            if curr_time - self.time_detected < Detector._absent_time_th:
                # nothing is detected, but there were detections not long ago
                #print(f"extended detection at {curr_time}")
                curr_detections = self.detections
            else:
                # nothing is detected for a long enough time, remove all saved detections
                if len(self.detections) > 0:
                    #print("absent")
                    self.detections = list()
                    self.time_detected = -1.0

        if len(curr_detections) > 0:
            # something is detected and verified
            self.detections = curr_detections

        if is_something_detected and self.time_detected < 0.0:
            # something is detected
            #print(f"initial detection at {curr_time}")
            self.time_detected = curr_time

        return curr_detections


if __name__ == "__main__":
    # Get configuration data from configuration yaml file
    config_yaml = "object_detection_video_stream.yaml"
    with open(config_yaml, "r") as file:
        config_data = yaml.safe_load(file)

    # Set all config options
    hw_location = config_data["hw_location"]
    model_zoo_url = config_data["model_zoo_url"]
    model_name = config_data["model_name"]
    video_source = config_data["video_source"]

    # Detection parameters
    labels: set = {"cat", "racoon", "dog"}
    time_th = config_data["time_th"]
    score_th = config_data["score_th"]

    # load object detection AI model
    model = dg.load_model(
        model_name=model_name,
        inference_host_address=hw_location,
        zoo_url=model_zoo_url,
        token=degirum_tools.get_token(),
    )

    # Create Detector
    detector = Detector(labels, score_th, time_th)

    # run AI inference on video stream
    inference_results = degirum_tools.predict_stream(model, video_source)

    # display inference results
    # Press 'x' or 'q' to stop
    with degirum_tools.Display("AI Camera") as display:
        for inference_result in inference_results:
            detections = detector.get_detections(inference_result)
            if len(detections) != 0:
                # send notification
                print("detected")
            display.show(inference_result)
            time.sleep(0.03)

