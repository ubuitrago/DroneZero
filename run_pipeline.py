"""
Entrypoint for the drone control pipeline.
This script uses ZSLAgent and FlightPlanAgent to process inputs and generate a flight plan.
"""

import argparse
import json
from typing import Optional, List
from PIL import Image
from zsl_agent import ZSLAgent
from flight_plan_agent import FlightPlanAgent
from experiment import Experiment
from drone_control import DroneController
import logging
import datetime

def run_pipeline(text_input: str, image_input: Optional[Image.Image] = None, sensor_data: Optional[List[float]] = None) -> dict:
    """
    Run the pipeline to generate a flight plan based on inputs.
    
    Args:
        text_input (str): Text description or command
        image_input (Optional[Image.Image]): Optional image input
        sensor_data (Optional[List[float]]): Optional sensor readings
        
    Returns:
        dict: Flight plan in the specified format
    """
    logging.basicConfig(level=logging.INFO, filename=f"pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log", filemode="a")
    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline with text: {text_input}")
    try:
        # Initialize agents
        zsl_agent = ZSLAgent()
        flight_plan_agent = FlightPlanAgent()
    
        # Analyze inputs using ZSLAgent
        zsl_output = zsl_agent.analyze_inputs(text_input=text_input, image_input=image_input, sensor_data=sensor_data)
        logger.info(f"ZSLAgent output: {zsl_output}\n\n")

        # Generate flight plan using FlightPlanAgent
        flight_plan = flight_plan_agent.generate_plan(zsl_output=zsl_output)
        logger.info(f"Flight plan: {flight_plan}\n\n")

        # Create an experiment instance
        experiment = Experiment()
        controller = DroneController()
        
        # Run the experiment
        experiment.run_experiment(controller=controller, flight_plan=flight_plan)
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return -1

    finally:
        controller.land()
        logger.info("Pipeline completed")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the drone control pipeline.")
    parser.add_argument("--text", type=str, required=True, help="Text description or command")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--sensor", type=str, help="Comma-separated list of sensor readings")
    
    args = parser.parse_args()
    
    # Process image input if provided
    image_input = None
    if args.image:
        image_input = Image.open(args.image)
    
    # Process sensor data if provided
    sensor_data = None
    if args.sensor:
        sensor_data = [float(x) for x in args.sensor.split(",")]
    
    # Run the pipeline
    result = run_pipeline(args.text, image_input, sensor_data)
    
    # Print the result
    print(json.dumps(result, indent=4)) 