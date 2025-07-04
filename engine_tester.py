#!/usr/bin/env python3
import numpy as np
import mujoco
from mujoco import MjModel, MjData
import mujoco_viewer

# MJCF definition of terrain, launch pad, and lander
XML = """
<mujoco>
  <asset>
    <hfield name="terrain" nrow="128" ncol="128" size="40 40 2 0.1"/>
    <material name="moon_soil" rgba="0.3 0.3 0.3 1"/>
    <material name="launch_pad" rgba="0.8 0.1 0.1 1"/>
    <material name="rocket_body" rgba="0.9 0.9 0.9 1"/>
    <material name="legs" rgba="0.2 0.2 0.2 1"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse="0.9 0.9 0.9" pos="0 0 10"/>
    <geom type="hfield" hfield="terrain" material="moon_soil" pos="0 0 0"/>
    <body pos="0 0 0.5">
      <geom type="cylinder" size="2 0.3" material="launch_pad"/>
    </body>
    <body name="lander" pos="0 0 15">
      <joint name="root" type="free"/>
      <geom type="cylinder" size="0.4 6" pos="0 0 0" material="rocket_body"/>
      <geom type="capsule" fromto="0 0 6 0 0 8" size="0.3" material="rocket_body"/>
      <geom type="capsule" fromto="0.6 0 -5.5 1.0 0 -8.0" size="0.1" material="legs"/>
      <geom type="capsule" fromto="-0.3 0.519 -5.5 -0.6 1.038 -8.0" size="0.1" material="legs"/>
      <geom type="capsule" fromto="-0.3 -0.519 -5.5 -0.6 -1.038 -8.0" size="0.1" material="legs"/>
      <geom type="sphere" size="0.15" pos="0.6 0 -5.5" material="legs"/>
      <geom type="sphere" size="0.15" pos="-0.3 0.519 -5.5" material="legs"/>
      <geom type="sphere" size="0.15" pos="-0.3 -0.519 -5.5" material="legs"/>
    </body>
  </worldbody>
</mujoco>
"""

def generate_terrain(model):
    """
    Procedurally generates a simple lunar heightmap and applies it to the model.
    """
    heightmap = np.zeros((128, 128), dtype=np.float32)
    for _ in range(5):
        x, y = np.random.randint(20, 108), np.random.randint(20, 108)
        size = np.random.uniform(5, 15)
        depth = np.random.uniform(0.5, 2.0)
        xx, yy = np.mgrid[:128, :128]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        heightmap += depth * np.exp(-dist ** 2 / (2 * (size ** 2)))
    heightmap += np.random.rand(128, 128) * 0.3
    heightmap[60:68, 60:68] = heightmap.min()
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    heightmap = heightmap * 1.5 + 0.1
    model.hfield_data[:] = heightmap.ravel()

def run_test(main=False, right=False, left=False, forward=False, back=False, steps=300):
    """
    Run a thrust test with specified engines fired continuously for a number of steps.

    :param main:    Fire the main engine (downward thrust)
    :param right:   Fire the right thruster (+x body axis)
    :param left:    Fire the left thruster (-x body axis)
    :param forward: Fire the forward thruster (+y body axis)
    :param back:    Fire the backward thruster (-y body axis)
    :param steps:   Number of simulation steps to run
    """
    # Load and initialize model & data
    model = MjModel.from_xml_string(XML)
    generate_terrain(model)
    data = MjData(model)
    mujoco.mj_resetData(model, data)

    # Viewer setup
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance  = 25.0
    viewer.cam.azimuth   = 35
    viewer.cam.elevation = -25

    # Build the combined body-frame thrust vector
    thrust_body = np.zeros(3)
    if main:
        thrust_body += np.array([  0.0,   0.0, -120.0])
    if right:
        thrust_body += np.array([ 40.0,   0.0,    0.0])
    if left:
        thrust_body += np.array([-40.0,   0.0,    0.0])
    if forward:
        thrust_body += np.array([  0.0,  40.0,    0.0])
    if back:
        thrust_body += np.array([  0.0, -40.0,    0.0])

    # Simulation loop
    for _ in range(steps):
        # Compute world-frame thrust via rotation matrix
        rot_flat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_flat, data.body(b"lander").xquat)
        rot = rot_flat.reshape(3, 3)
        world_thrust = rot @ thrust_body

        # Apply force at the center of mass
        mujoco.mj_applyFT(
            model, data,
            world_thrust,                  # force vector
            np.zeros(3),                   # torque (none)
            np.zeros(3),                   # point (center of mass)
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, b"lander"
            ),
            data.qfrc_applied
        )

        mujoco.mj_step(model, data)
        viewer.render()

    viewer.close()

# Example usage when run as a script
if __name__ == "__main__":
    # Test all five engines for 300 steps
    run_test(main=True, right=True, left=True, forward=True, back=True)