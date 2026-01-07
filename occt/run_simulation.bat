@echo off
chcp 65001 >nul 2>&1
echo ==============================================
echo          OCCT离线仿真一键启动脚本
echo ==============================================
echo.

:: ===================== 请修改以下配置（仅需改这部分） =====================
:: 1. Python解释器路径（建议用conda环境的python）
set PYTHON_PATH=D:\miniconda3\envs\occt_haolin\python.exe
:: 2. 离线仿真脚本路径
set SIM_SCRIPT_PATH=offline_simulation.py
:: 3. Checkpoint文件路径（必填！修改为你的实际路径）
set CKPT_PATH="E:\rl\occt\outputs\2026-01-06\19-06-31\checkpoints_occt\checkpoint_1024000_frames.pt"
:: 4. 仿真轮数
set NUM_EPISODES=5
:: 5. 是否启用可视化（true/false）
set ENABLE_VISUALIZATION=true
:: 6. 是否保存视频（true/false，需先启用可视化）
set SAVE_VIDEO=true
:: 7. 视频保存目录（留空则默认保存在Checkpoint同目录）
set VIDEO_DIR=""
:: 8. 运行设备（cpu/cuda:0）
set DEVICE="cpu"
:: =========================================================================

echo [配置信息]
echo Python解释器：%PYTHON_PATH%
echo 仿真脚本：%SIM_SCRIPT_PATH%
echo Checkpoint路径：%CKPT_PATH%
echo 仿真轮数：%NUM_EPISODES%
echo 可视化：%ENABLE_VISUALIZATION%
echo 保存视频：%SAVE_VIDEO%
echo 视频目录：%VIDEO_DIR%
echo 运行设备：%DEVICE%
echo.

:: 构建运行命令
set RUN_CMD=%PYTHON_PATH% %SIM_SCRIPT_PATH% --ckpt_path %CKPT_PATH% --num_episodes %NUM_EPISODES% --device %DEVICE%

:: 判断是否启用可视化
if /i "%ENABLE_VISUALIZATION%"=="true" (
    set RUN_CMD=%RUN_CMD% --enable_visualization
)

:: 判断是否保存视频
if /i "%SAVE_VIDEO%"=="true" (
    set RUN_CMD=%RUN_CMD% --save_video
)

:: 判断是否指定视频目录
if not %VIDEO_DIR%=="" (
    set RUN_CMD=%RUN_CMD% --video_dir %VIDEO_DIR%
)

echo [运行命令] %RUN_CMD%
echo.
echo 开始运行离线仿真...
echo ===================== 运行日志 =====================
echo.

:: 执行命令
%RUN_CMD%

echo.
echo ===================== 运行结束 =====================
echo 按任意键退出...
pause >nul