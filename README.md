# CuiYangxiang_convert_hdf5_to_lerobot_v3
convert_hdf5_to_lerobot_v3.py` 用于把 HDF5 数据文件转换为 LeRobot v3.0 可读取的数据集格式。脚本会读取 HDF5 中的时间序列数据，并按参数映射为 LeRobot 的核心字段，例如 `observation.state` 和 `action`。常见 ALOHA 风格数据可自动识别；其他数据需要手动指定字段路径。  运行前请先进入项目目录并激活虚拟环境.转换完成后，结果会保存在 `--output` 指定目录中。使用前应确认 HDF5 中哪些字段是真实状态，哪些字段是真实动作。
