#!/usr/bin/env python3

import sys
import os
# Add parent directory to path so we can import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from environment.simple import SimplePathFollowingEnv
import time

def test_trailer_connection():
    """
    Automatic test to validate trailer connection
    Executes all scenarios automatically without user interaction
    """
    env = SimplePathFollowingEnv()
    obs = env.reset()
    
    # List of tests with specific movements
    test_scenarios = [
        {
            "name": "1. Straight line movement",
            "actions": [(0.8, 0.0)] * 100,
            "description": "Basic test - trailer should follow in straight line"
        },
        {
            "name": "2. Right turn",
            "actions": [(0.6, -0.8)] * 15,
            "description": "Trailer should make tighter curve than agent"
        },
        {
            "name": "3. Left turn", 
            "actions": [(0.6, 0.8)] * 15,
            "description": "Trailer should make tighter curve than agent"
        },
        {
            "name": "4. S-curve movement",
            "actions": [(0.5, 0.6)] * 8 + [(0.5, -0.6)] * 8 + [(0.5, 0.6)] * 8,
            "description": "Agility test - direction changes"
        },
        {
            "name": "5. Stop and reverse",
            "actions": [(0.0, 0.0)] * 5 + [(-0.4, 0.0)] * 10,
            "description": "Reverse test - trailer should follow correctly"
        },
        {
            "name": "6. Complex maneuver",
            "actions": [(0.7, 0.0)] * 5 + [(0.5, 1.0)] * 10 + [(0.8, 0.0)] * 5 + [(0.5, -1.0)] * 10,
            "description": "Combination of movements"
        }
    ]
    
    # Initialize single plot window
    plt.ion()
    if not hasattr(env, 'fig'):
        env.render()  # Create the initial figure
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        time.sleep(1)  # Brief pause
        
        distances = []
        
        for i, action in enumerate(scenario['actions']):
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Calculate metrics
            distance = np.linalg.norm(env.current_position - env.trailer_real_position)
            distances.append(distance)
            
            # Render every 5 steps
            if i % 5 == 0:
                env.render()
                plt.pause(0.05)
            
            if terminated or truncated:
                obs = env.reset()
                break
        
        # Simple validation
        max_distance = np.max(distances)
        if max_distance > 1.2:
            print("FAILURE")
        else:
            print("SUCCESS")
    
    plt.ioff()
    print("\nTest completed!")

if __name__ == "__main__":
    test_trailer_connection()
