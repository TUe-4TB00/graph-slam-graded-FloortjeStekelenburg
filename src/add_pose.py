
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):

      # Relative motion from X(3) to X(4)
    odometry = gtsam.Pose2(2.0,0.0, math.radians(90))

    # Add odometry factor
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # Compute initial estimate for X(4)
    prev_pose = initial_estimate.atPose2(X(3))
    new_pose = prev_pose.compose(odometry)

    # Add initial estimate
    initial_estimate.insert(X(4), new_pose)


    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2)
    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
    
    return graph, initial_estimate