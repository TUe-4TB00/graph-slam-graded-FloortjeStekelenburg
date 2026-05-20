import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    # TODO: Perform the optimization and print the result

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate) 
    result = optimizer.optimize() 
    #print(result)

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    import matplotlib.pyplot as plt
    import gtsam.utils.plot as gp
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = 'd'     # chosen pose option
    best_landmark = 1   # chosen landmark (1 or 2)

    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    marginals = gtsam.Marginals(graph, result)
    
    fig = plt.figure(2)
    axes = fig.add_subplot()
    axes = fig.axes[0]

    # Plot 2D poses
    poses = gtsam.utilities.allPose2s(result)
    for key in poses.keys():
        pose = poses.atPose2(key)
        covariance = marginals.marginalCovariance(key)

        gp.plot_pose2_on_axes(axes, pose, covariance=covariance, axis_length=0.3)

    # Plot 2D landmarks
    landmarks: np.ndarray = gtsam.utilities.extractPoint2(result)  # 2xn array
    for j, landmark in enumerate(landmarks):
        gp.plot_point2_on_axes(axes, landmark, linespec="b")
        covariance = marginals.marginalCovariance(L(j+1))
        gp.plot_covariance_ellipse_2d(axes, landmark, covariance=covariance)

    axes.set_aspect("equal", adjustable="datalim")

    
    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    sum_of_marginals = (marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum())
    print(f"Sum of marginals: {sum_of_marginals}")
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = 'd'
    best_landmark = 1

    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    pose1 = result.atPose2(X(1))
    pose2 = result.atPose2(X(2))
    pose3 = result.atPose2(X(3))

    true_pose1 = gtsam.Pose2(0.0, 0.0, 0.0)
    true_pose2 = gtsam.Pose2(2.0, 0.0, 0.0)
    true_pose3 = gtsam.Pose2(4.0, 0.0, 0.0)

    error_x1 = abs(pose1.x() - true_pose1.x())
    error_y1 = abs(pose1.y() - true_pose1.y())
    error_theta1 = abs(pose1.theta() - true_pose1.theta())
    error_x2 = abs(pose2.x() - true_pose2.x())
    error_y2 = abs(pose2.y() - true_pose2.y())
    error_theta2 = abs(pose2.theta() - true_pose2.theta())
    error_x3 = abs(pose3.x() - true_pose3.x())
    error_y3 = abs(pose3.y() - true_pose3.y())
    error_theta3 = abs(pose3.theta() - true_pose3.theta())


    list_of_errors = [
    error_x1, error_y1, error_theta1,
    error_x2, error_y2, error_theta2,
    error_x3, error_y3, error_theta3]
    # TODO: compute the sum of the errors and return it along with the best pose and landmark

    sum_of_errors = sum(list_of_errors)

    print(f"sum of errors: {sum_of_errors}")
    
    
    return best_pose, best_landmark, sum_of_errors 