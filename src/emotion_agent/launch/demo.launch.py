from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1) 先用 stub 模拟情绪模型输出（你后面替换成真实模型）
        Node(
            package="emotion_agent",
            executable="emotion_stub",
            name="emotion_stub",
            output="screen",
            parameters=[{
                "publish_hz": 0.5,   # 每2秒换一次情绪，方便你看
            }]
        ),

        # 2) bandit agent：订阅 emotion+credibility，输出 face_id + body_id
        Node(
            package="emotion_agent",
            executable="bandit_agent",
            name="bandit_agent",
            output="screen",
            parameters=[{
                "credibility_threshold": 0.55,  # 低于阈值走保守动作集合
                "epsilon": 0.25,                # 探索率
                "alpha": 0.25,                  # 学习率（用于简单增量更新）
                "log_path": "/tmp/emotion_agent_log.jsonl",
            }]
        ),

        # 3) face publisher：把 face_id 发布成图片（rqt_image_view 看）
        Node(
            package="emotion_agent",
            executable="face_publisher",
            name="face_publisher",
            output="screen",
        ),

        # 4) expression controller：把 body_id 转成 /cmd_vel（Gazebo/真机都通用）
        Node(
            package="emotion_agent",
            executable="expression_controller",
            name="expression_controller",
            output="screen",
            parameters=[{
                "cmd_vel_topic": "/model/tugbot/cmd_vel",
                "action_duration_sec": 1.2
            }]
        ),
    ])