from setuptools import find_packages, setup

package_name = "emotion_agent"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/demo.launch.py"]),
        ("share/" + package_name + "/assets/faces", [
            "assets/faces/neutral.png",
            "assets/faces/happy.png",
            "assets/faces/sad.png",
            "assets/faces/angry.png",
            "assets/faces/excited.png",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Emotion-aware robot agent with contextual bandit learning",
    license="MIT",
    entry_points={
        "console_scripts": [
            "emotion_stub = emotion_agent.emotion_stub:main",
            "bandit_agent = emotion_agent.bandit_agent:main",
            "expression_controller = emotion_agent.expression_controller:main",
            "reward_keyboard = emotion_agent.reward_keyboard:main",
            "face_publisher = emotion_agent.face_publisher:main",
            "gz_simple_bot_controller = emotion_agent.gz_simple_bot_controller:main",
            "gz_avatar_motion = emotion_agent.gz_avatar_motion:main",
            'emotion_model_node = emotion_agent.emotion_model_node:main',
        ],
    },
)