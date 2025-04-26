"""
This module is based on and extends the functionality of the original airsim_wrapper.py
from the PromptCraft-Robotics GitHub repository (https://github.com/PromptCraft-Robotics).

Original work Copyright (c) PromptCraft-Robotics
Modifications Copyright (c) 2025 Uriel Buitrago

This work is licensed under the same terms as the original PromptCraft-Robotics repository.
For more information, see the original repository's LICENSE file.

The modifications include:
- Enhanced drone control capabilities
- Additional orientation control methods
- Integration with experiment recording
- Custom flight path planning
"""

import airsim
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

objects_dict = {
    "turbine1": "BP_Wind_Turbines_C_1",
    "turbine2": "StaticMeshActor_2",
    "solarpanels": "StaticMeshActor_146",
    "crowd": "StaticMeshActor_6",
    "car": "StaticMeshActor_10",
    "tower1": "SM_Electric_trellis_179",
    "tower2": "SM_Electric_trellis_7",
    "tower3": "SM_Electric_trellis_8",
}

class DroneController:
    def __init__(self, ip: str = "127.0.0.1", port: int = 41451):
        """
        Initialize the drone controller with AirSim connection parameters.
        
        Args:
            ip (str): IP address of the AirSim server
            port (int): Port number of the AirSim server
        """
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self._home_position = None  # Will store the takeoff position
        
    def takeoff(self, altitude: float = 5.0) -> bool:
        """
        Take off to a specified altitude.
        
        Args:
            altitude (float): Target altitude in meters
            
        Returns:
            bool: True if takeoff was successful
        """
        try:
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(-altitude, 1).join()
            # Store the current position as home position after takeoff
            self._home_position = self.get_position()
            return True
        except Exception as e:
            print(f"Takeoff failed: {str(e)}")
            return False
            
    def land(self) -> bool:
        """
        Land the drone safely.
        
        Returns:
            bool: True if landing was successful
        """
        try:
            self.client.landAsync().join()
            return True
        except Exception as e:
            print(f"Landing failed: {str(e)}")
            return False
            
    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float) -> bool:
        """
        Move the drone by velocity for a specified duration.
        
        Args:
            vx (float): Velocity in x direction (m/s)
            vy (float): Velocity in y direction (m/s)
            vz (float): Velocity in z direction (m/s)
            duration (float): Duration of movement in seconds
            
        Returns:
            bool: True if movement was successful
        """
        try:
            self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
            return True
        except Exception as e:
            print(f"Movement failed: {str(e)}")
            return False
            
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get current position of the drone.
        
        Returns:
            Tuple[float, float, float]: (x, y, z) position in meters
        """
        position = self.client.getMultirotorState().kinematics_estimated.position
        return (position.x_val, position.y_val, position.z_val)
        
    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Get current velocity of the drone.
        
        Returns:
            Tuple[float, float, float]: (vx, vy, vz) velocity in m/s
        """
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return (velocity.x_val, velocity.y_val, velocity.z_val)
    
    def set_yaw(self, yaw: float) -> None:
        """
        Set internal yaw of the drone.
        
        Args:
            yaw (float): Target yaw angle in degrees
            
        Returns:
            None
        """
        self.client.rotateToYawAsync(yaw, 5).join()
        pass

    def get_yaw(self) -> float:
        """
        Get current yaw of the drone.

        Returns:
            float: yaw in degrees
        """
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw
        
    def get_pitch(self) -> float:
        """
        Get current pitch of the drone.
        
        Returns:
            float: pitch in degrees
        """
        orientation_quat = self.client.simGetVehiclePose().orientation
        pitch = airsim.to_eularian_angles(orientation_quat)[1]
        return pitch
        
    def get_roll(self) -> float:
        """
        Get current roll of the drone.
        
        Returns:
            float: roll in degrees
        """
        orientation_quat = self.client.simGetVehiclePose().orientation
        roll = airsim.to_eularian_angles(orientation_quat)[0]
        return roll
        
    def set_orientation(self, pitch: float, roll: float, yaw: float) -> None:
        """
        Set the complete orientation of the drone.
        
        Args:
            pitch (float): Target pitch angle in degrees
            roll (float): Target roll angle in degrees
            yaw (float): Target yaw angle in degrees
            
        Returns:
            None
        """
        # Convert degrees to quaternion
        orientation_quat = airsim.to_quaternion(roll, pitch, yaw)
        pose = self.client.simGetVehiclePose()
        pose.orientation = orientation_quat
        self.client.simSetVehiclePose(pose, True)
        
    def get_orientation(self) -> Tuple[float, float, float]:
        """
        Get the complete orientation of the drone.
        
        Returns:
            Tuple[float, float, float]: (roll, pitch, yaw) in degrees
        """
        orientation_quat = self.client.simGetVehiclePose().orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation_quat)
        return (roll, pitch, yaw)
        
    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()
        
    def return_to_home(self, altitude: Optional[float] = None) -> bool:
        """
        Return the drone to its takeoff position (home position).
        
        Args:
            altitude (Optional[float]): If specified, the drone will first move to this altitude
                                      before returning to home. If None, uses current altitude.
            
        Returns:
            bool: True if return to home was successful
        """
        if self._home_position is None:
            print("Error: Home position not set. Takeoff must be performed first.")
            return False
            
        try:
            current_pos = self.get_position()
            
            # If altitude is specified, move to that altitude first
            if altitude is not None:
                self.client.moveToZAsync(-altitude, 1).join()
            
            # Move to home position at current altitude
            target_pos = airsim.Vector3r(
                self._home_position[0],
                self._home_position[1],
                current_pos[2]  # Maintain current altitude
            )
            
            # Move to home position
            self.client.moveToPositionAsync(
                target_pos.x_val,
                target_pos.y_val,
                target_pos.z_val,
                1.0  # velocity in m/s
            ).join()
            
            return True
        except Exception as e:
            print(f"Return to home failed: {str(e)}")
            return False
        
    def execute_gesture_command(self, gesture_data: Dict) -> bool:
        """
        Execute a drone command based on gesture data.
        
        Args:
            gesture_data (Dict): Dictionary containing gesture interpretation
            
        Returns:
            bool: True if command was executed successfully
        """
        # This method will be expanded based on your gesture recognition system
        # For now, it's a placeholder that can be customized
        try:
            # Example implementation - to be modified based on your gesture system
            if gesture_data.get("command") == "move_forward":
                return self.move_by_velocity(1.0, 0.0, 0.0, 1.0)
            elif gesture_data.get("command") == "move_backward":
                return self.move_by_velocity(-1.0, 0.0, 0.0, 1.0)
            # Add more gesture commands as needed
            return False
        except Exception as e:
            print(f"Gesture command execution failed: {str(e)}")
            return False
            
    def start_recording(self) -> None:
        """
        Start recording camera images.
        """
        self.client.startRecording()
        
    def stop_recording(self) -> None:
        """
        Stop recording camera images.
        """
        self.client.stopRecording()
        
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
                # Get the actual image dimensions from the response
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                # Calculate the expected size based on height and width
                expected_size = response.height * response.width * 3
                if len(img1d) == expected_size:
                    img = img1d.reshape(response.height, response.width, 3)
                else:
                    # If size doesn't match, try to handle it gracefully
                    print(f"Warning: Image size mismatch for camera {idx}. Expected {expected_size}, got {len(img1d)}")
                    continue
            images[f"camera_{idx}"] = img
            
        return images
            
    def __del__(self):
        """
        Cleanup when the controller is destroyed.
        """
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass 