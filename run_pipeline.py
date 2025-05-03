"""
Entrypoint for the drone control pipeline.
This script uses ZSLAgent and FlightPlanAgent to process inputs and generate a flight plan.
"""

import argparse
import json
import os
import numpy as np
from typing import Optional, List
from PIL import Image
from zsl_agent import ZSLAgent
from flight_plan_agent import FlightPlanAgent
from experiment import Experiment
from drone_control import DroneController
import logging
import datetime
from dotenv import load_dotenv

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_file = os.path.join(log_dir, f"pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will output to console as well
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created at: {log_file}")
    return logger

def load_prediction_data(prediction_dir: str) -> dict:
    """
    Load prediction data from the input_predict.py output directory.
    
    Args:
        prediction_dir (str): Path to the prediction output directory
        
    Returns:
        dict: Dictionary containing all prediction data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading prediction data from: {prediction_dir}")
    
    if not os.path.exists(prediction_dir):
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
        
    # Load prediction results
    prediction_path = os.path.join(prediction_dir, 'prediction.txt')
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    
    # Read the entire prediction file
    with open(prediction_path, "r") as pp:
        prediction_text = pp.read()
    
     # Load ngrok URL from .env file
    env_path = os.path.join(prediction_dir, '.env')
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment file not found at: {env_path}")
    load_dotenv(env_path)
    gif_url = os.getenv('URL')
    if not gif_url:
        raise ValueError("URL not found in environment file")
    
    # # Load annotated frame
    # gif_path = os.path.join(prediction_dir, 'prediction.gif')
    # if not os.path.exists(gif_path):
    #     raise FileNotFoundError(f"Gif file not found: {gif_path}")
    
    logger.info("Successfully loaded all prediction data")
    return {
        'prediction': prediction_text,
        'gif': gif_url
    }

def run_pipeline(prediction_dir: str) -> dict:
    """
    Run the pipeline to generate a flight plan based on gesture prediction data.
    
    Args:
        prediction_dir (str): Path to the prediction output directory from input_predict.py
        
    Returns:
        dict: Flight plan in the specified format
    """
    # NOTE: For the demo Video, we copied the ChatDocument output from each agent and created static variables. 
    # The variables are named as ZSL_DEMO_DICT and FP_AGENT_DEMO_DICT.
    # The actual implementation of the agents is in the zsl_agent.py and flight_plan_agent.py files.
    # For the actual implementation, please refer to the files. 
    # TODO: Fix the errors with Langroid.Task returning None without waiting for the ChatDocument to complete. 
    logger = setup_logging()
    
    # Initialize controller early to ensure it's always available
    controller = DroneController()
    
    try:
        # Load prediction data
        prediction_data = load_prediction_data(prediction_dir)
        logger.info(f"Loaded prediction data from {prediction_dir}")
        
        # Initialize agents
        zsl_agent = ZSLAgent()
        flight_plan_agent = FlightPlanAgent()
        
        # Convert keypoints to list format if it's a numpy array
        # keypoints_data = prediction_data['keypoints']
        # if isinstance(keypoints_data, np.ndarray):
        #     keypoints_data = keypoints_data.tolist()
        
        # Analyze inputs using ZSLAgent
        zsl_output = zsl_agent.analyze_inputs(
            text_input=prediction_data['prediction'],
            image_input=prediction_data['gif'],
        )
        logger.info(f"ZSLAgent output: {zsl_output}\n\n")
        ZSL_DEMO_DICT = {
    "predicted_gesture": "move ahead (fly forward at moderate speed)",
    "reasoning": "The majority of detected gestures in the sequence are 'Move Ahead', characterized by arms extended horizontally to the shoulders, hands above eye level, palms facing backward, and a beckoning motion. This is a classic aviation and ground crew signal for 'proceed forward'. The intermittent 'All Clear' signals confirm there are no collision hazards, supporting a safe forward flight. The rapidity of the beckoning suggests a moderate, not slow, movement is desired.",
    "insight": "For a deaf person, this gesture would be interpreted as a clear directional command: the repeated arm sweep backward (beckoning) is a widely recognized visual cue for 'come forward' or 'move ahead'. The sequence of 'All Clear' gestures further assures the drone that it is safe to proceed. Flight planning insight: Maintain current altitude (12ft), move forward in the direction the person is facing, and adjust speed to match the rapidity of the beckoning arms. Remain alert for future gesture changes."
}
        # Generate flight plan using FlightPlanAgent
        flight_plan = flight_plan_agent.generate_plan(zsl_agent_output=ZSL_DEMO_DICT)
        FP_AGENT_DEMO_DICT = {
            "flight_plan": [
                ("takeoff", (3.66,)),  # 12 feet converted to meters is approximately 3.66 m
                ("move_by_velocity", (1.5, 0.0, 0.0, 4.0)),  # Move forward at moderate speed (1.5 m/s) for 4 seconds
                ("land", ())
            ]
        }
        DEMO_LIST = FP_AGENT_DEMO_DICT['flight_plan']
        logger.info(f"Flight plan: {flight_plan}\n\n")

        # Create an experiment instance
        experiment = Experiment()
        
        # Run the experiment
        experiment.run_experiment(controller=controller, flight_plan=DEMO_LIST)
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        return -1

    finally:
        # Ensure controller is landed in case of any errors
        try:
            controller.land()
        except Exception as e:
            logger.error(f"Error during landing: {e}", exc_info=True)
        logger.info("Pipeline completed")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the drone control pipeline.")
    parser.add_argument("--prediction_dir", type=str, required=True, 
                       help="Path to the prediction output directory from input_predict.py")
    
    args = parser.parse_args()
    
    # Run the pipeline
    result = run_pipeline(args.prediction_dir)
    
    # Print the result
    print(json.dumps(result, indent=4)) 