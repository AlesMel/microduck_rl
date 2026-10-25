"""1v1 duck football: a high-level policy that steers the shipped walker.

The match policy never touches a joint. It outputs a twist (and later a kick trigger) and the
frozen `velstand.onnx` — the walk a real duck ships with — turns that into joint targets at 50 Hz.
That keeps the learning problem at the level Unity's soccer agents play at (where to go, when to
push the ball), and it is the same split robotd has on a robot: `robot.move` in, policy out.
"""
