# Moorebot RL MVP — build something tonight

This is a lightweight **2-D simulated** goal navigation project using PPO. It does **not** require Gazebo or a robot to train. This is a preliminary policy, not a ready-to-deploy Moorebot controller.

## Fast start (Ubuntu x86_64 / ROS 2 Humble laptop)

```bash
cd moorebot_rl_mvp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_rl.txt
python demo.py --random --episodes 2 --render
python train.py --steps 100000
python demo.py --episodes 10 --render
```

If using ROS 2 Humble system Python, train using the venv above, but run ROS nodes via system `/usr/bin/python3` (or a compatible venv with ROS packages installed). The simulation doesn't need ROS.

## What this trains

- 6m by 6m room with walls and two partial dividers
- Random start and goal
- 9 synthetic 270-degree range rays plus relative goal distance/bearing and previous action
- Continuous action: forward speed and rotation
- Collision penalty, progress reward, goal bonus
- Outputs `models/moorebot_ppo.zip`

## Important before real deployment

The Moorebot starter bridge exposes `/ROBOT/camera/image`, `/ROBOT/sensor/tof`, `/ROBOT/sensor/imu`, and `/ROBOT/cmd_vel`. The simulation uses **nine synthetic range rays** that the real Scout does NOT provide. It also assumes true goal-relative location, which the original code does NOT supply. Consequently **do not load this model directly onto the real robot**. To deploy, you must either (a) design real camera-based free-space/depth features equivalent to these observations and a robot localization source, then retrain with realistic noise, or (b) retrain a policy with the robot's actually available observations. Add an external emergency stop and enforce speed bounds.

The environment dynamics are illustrative and don't model Scout mecanum actuation, friction, camera latency, or sensor noise. There is no guarantee PPO will learn reliable navigation within 100k steps; evaluate performance and improve training accordingly.

## Hardware bridge smoke tests (separate from RL)

Your provided `start_bridge.sh` can be used after connecting to the Scout hotspot. Read the original starter README first. Check `ros2 topic list`, `ros2 topic echo /ROBOT/sensor/tof`, and `ros2 topic hz /ROBOT/camera/image`. Test movement only in a clear area with a physical stop method.
