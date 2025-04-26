import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from drone_control import DroneController

class Experiment:
    def __init__(self, 
                 experiment_name: str = "",
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
        # Setup recording parameters
        self.record_images = record_images
        self.record_telemetry = record_telemetry
        
        # Create experiment directory
        if not experiment_name:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize data storage
        self.telemetry_data = []
        self.start_time = None
        
    def start_recording(self, controller: DroneController) -> None:
        """
        Start recording experiment data.
        
        Args:
            controller (DroneController): The drone controller to use for recording
        """
        self.start_time = time.time()
        
        if self.record_images:
            controller.start_recording()
            
        if self.record_telemetry:
            self.telemetry_data = []
            
    def stop_recording(self, controller: DroneController) -> None:
        """
        Stop recording and save all experiment data.
        
        Args:
            controller (DroneController): The drone controller to use for recording
        """
        if self.record_images:
            controller.stop_recording()
            
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
            
    # def save_camera_images(self, controller: DroneController) -> None:
    #     """
    #     Save current camera images to the experiment directory.
        
    #     Args:
    #         controller (DroneController): The drone controller to use for getting images
    #     """
    #     images = controller.get_camera_images()
    #     for camera_name, img in images.items():
    #         img_file = os.path.join(self.save_dir, f"{camera_name}_{time.time()}.png")
    #         airsim.write_png(img_file, img)
            
    def run_experiment(self, 
                      controller: DroneController,
                      flight_plan: List[Tuple[str, tuple]],
                      additional_data_callback: Optional[callable] = None) -> None:
        """
        Run a complete experiment with the given flight plan.
        
        Args:
            controller (DroneController): The drone controller to use
            flight_plan (List[Tuple[str, tuple]]): List of (method_name, args) tuples where:
                - method_name (str): Name of the DroneController method to call
                - args (tuple): Arguments to pass to the method
            additional_data_callback (Optional[callable]): Function to get additional data to record
        """
        try:
            # Start recording before any commands
            self.start_recording(controller)
            
            # Execute each command in the flight plan
            for method_name, args in flight_plan:
                # Get the method from the controller
                method = getattr(controller, method_name)
                
                # Call the method with its arguments
                if isinstance(args, tuple):
                    method(*args)
                else:
                    method(args)
                
                # Record data continuously
                position = controller.get_position()
                orientation = controller.get_orientation()
                velocity = controller.get_velocity()
                
                additional_data = None
                if additional_data_callback:
                    additional_data = additional_data_callback()
                    
                self.record_telemetry_frame(
                    position, orientation, velocity, additional_data
                )
                
                # if self.record_images:
                #     self.save_camera_images(controller)
            
        finally:
            # Stop recording after all commands are complete
            self.stop_recording(controller)

if __name__ == "__main__":
    # Create experiment instance
    # experiment = Experiment(experiment_name="test_experiment")
    experiment = Experiment()
    
    # Create drone controller
    controller = DroneController()
    
    # Define a simple flight plan
    flight_plan = [
        ("takeoff", (5.0,)),  # Take off to 5 meters
        ("move_by_velocity", (1.0, 0.0, 0.0, 2.0)),  # Move forward at 1 m/s for 2 seconds
        ("move_by_velocity", (-1.0, 0.0, 0.0, 2.0)),  # Move backward at 1 m/s for 2 seconds
        ("land", ())  # Land safely
    ]
    
    # Run the experiment
    experiment.run_experiment(controller, flight_plan)
    
