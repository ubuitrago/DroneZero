import airsim
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from drone_control import DroneController

class Experiment:
    def __init__(self, 
                 experiment_name: str = None,
                 save_dir: str = "experiment_data",
                 record_images: bool = True,
                 record_telemetry: bool = True):
        """
        Initialize an experiment for recording drone flight data.
        
        Args:
            experiment_name (str): Name of the experiment. If None, will use timestamp
            save_dir (str): Directory to save experiment data
            record_images (bool): Whether to record camera images
            record_telemetry (bool): Whether to record telemetry data
        """
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Setup recording parameters
        self.record_images = record_images
        self.record_telemetry = record_telemetry
        
        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize data storage
        self.telemetry_data = []
        self.start_time = None
        
    def start_recording(self) -> None:
        """
        Start recording experiment data.
        """
        self.start_time = time.time()
        
        if self.record_images:
            # Start recording images from all cameras
            self.client.startRecording()
            
        if self.record_telemetry:
            # Initialize telemetry data list
            self.telemetry_data = []
            
    def stop_recording(self) -> None:
        """
        Stop recording and save all experiment data.
        """
        if self.record_images:
            self.client.stopRecording()
            
        if self.record_telemetry:
            self._save_telemetry_data()
            
    def record_telemetry_frame(self, 
                             position: Tuple[float, float, float],
                             orientation: Tuple[float, float, float],
                             velocity: Tuple[float, float, float],
                             additional_data: Optional[Dict] = None) -> None:
        """
        Record a single frame of telemetry data.
        
        Args:
            position (Tuple[float, float, float]): (x, y, z) position
            orientation (Tuple[float, float, float]): (roll, pitch, yaw) angles
            velocity (Tuple[float, float, float]): (vx, vy, vz) velocity
            additional_data (Optional[Dict]): Any additional data to record
        """
        if not self.record_telemetry:
            return
            
        frame_data = {
            "timestamp": time.time() - self.start_time,
            "position": {
                "x": position[0],
                "y": position[1],
                "z": position[2]
            },
            "orientation": {
                "roll": orientation[0],
                "pitch": orientation[1],
                "yaw": orientation[2]
            },
            "velocity": {
                "vx": velocity[0],
                "vy": velocity[1],
                "vz": velocity[2]
            }
        }
        
        if additional_data:
            frame_data.update(additional_data)
            
        self.telemetry_data.append(frame_data)
        
    def _save_telemetry_data(self) -> None:
        """
        Save recorded telemetry data to a JSON file.
        """
        if not self.telemetry_data:
            return
            
        telemetry_file = os.path.join(self.save_dir, "telemetry.json")
        with open(telemetry_file, 'w') as f:
            json.dump(self.telemetry_data, f, indent=4)
            
    def get_camera_images(self) -> Dict[str, np.ndarray]:
        """
        Get images from all cameras.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of camera names and their images
        """
        images = {}
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene),
            airsim.ImageRequest("1", airsim.ImageType.Scene)
        ])
        
        for idx, response in enumerate(responses):
            if response.pixels_as_float:
                img = np.array(response.image_data_float, dtype=np.float32)
                img = img.reshape(response.height, response.width)
            else:
                img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img = img.reshape(response.height, response.width, 3)
            images[f"camera_{idx}"] = img
            
        return images
        
    def save_camera_images(self) -> None:
        """
        Save current camera images to the experiment directory.
        """
        images = self.get_camera_images()
        for camera_name, img in images.items():
            img_file = os.path.join(self.save_dir, f"{camera_name}_{time.time()}.png")
            airsim.write_png(img_file, img)
            
    def run_experiment(self, 
                      controller: DroneController,
                      flight_plan: List[Tuple[float, float, float]],
                      additional_data_callback: Optional[callable] = None) -> None:
        """
        Run a complete experiment with the given flight plan.
        
        Args:
            controller (DroneController): The drone controller to use
            flight_plan (List[Tuple[float, float, float]]): List of (x, y, z) waypoints
            additional_data_callback (Optional[callable]): Function to get additional data to record
        """
        try:
            self.start_recording()
            
            # Takeoff
            controller.takeoff()
            
            # Follow flight plan
            for waypoint in flight_plan:
                # Move to waypoint
                controller.client.moveToPositionAsync(
                    waypoint[0], waypoint[1], -waypoint[2], 5
                ).join()
                
                # Record data
                position = controller.get_position()
                orientation = controller.get_orientation()
                velocity = controller.get_velocity()
                
                additional_data = None
                if additional_data_callback:
                    additional_data = additional_data_callback()
                    
                self.record_telemetry_frame(
                    position, orientation, velocity, additional_data
                )
                
                if self.record_images:
                    self.save_camera_images()
                    
            # Return to home and land
            controller.return_to_home()
            controller.land()
            
        finally:
            self.stop_recording() 