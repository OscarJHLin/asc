#!/usr/bin/env bash
# ============================================================
# Asc 服务器环境初始化脚本（一键编译，自动检测 GPU）
# 自动检测：NVIDIA CUDA / AMD ROCm / Apple Metal / CPU-only
# 无需 sudo，所有工具安装到用户目录
# 用法：bash setup_llama_cpp.sh [--accept-defaults]
# ============================================================

set -euo pipefail

# ---------- 配置 ----------
LLAMA_CPP_VERSION="b4683"
INSTALL_PREFIX="$HOME/.local"
ASC_PROJECT="$HOME/projects/asc"
CUDA_INSTALL_DIR="$HOME/.local/cuda"
ROCM_INSTALL_DIR="$HOME/.local/rocm"

ACCEPT_DEFAULTS=false
[[ "${1:-}" == "--accept-defaults" ]] && ACCEPT_DEFAULTS=true

# ============================================================
# 函数定义
# ============================================================

prompt_yesno() {
    local question="$1"
    local default="${2:-Y}"

    if $ACCEPT_DEFAULTS; then
        echo "${question} [${default}] -> ${default}"
        [[ "$default" == "Y" || "$default" == "y" ]]
        return $?
    fi

    local prompt_str
    if [[ "$default" == "Y" || "$default" == "y" ]]; then
        prompt_str="[Y/n]"
    else
        prompt_str="[y/N]"
    fi

    while true; do
        read -r -p "${question} ${prompt_str} " answer
        answer="${answer:-$default}"
        case "$answer" in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
            *) echo "  请输入 Y 或 N" ;;
        esac
    done
}

ensure_cmake() {
    # 检查常见路径
    for cmake_candidate in \
        "cmake" \
        "${ASC_PROJECT}/.venv/bin/cmake" \
        "$HOME/.local/bin/cmake"; do
        if command -v "$cmake_candidate" &>/dev/null; then
            cmake_dir=$(dirname "$(command -v "$cmake_candidate")")
            if ! echo "$PATH" | grep -q "$cmake_dir"; then
                export PATH="$cmake_dir:$PATH"
            fi
            return 0
        fi
    done

    # 检查 venv 内嵌 cmake
    local venv_cmake=""
    for f in "${ASC_PROJECT}/.venv/lib/python"*/site-packages/cmake/data/bin/cmake; do
        if [ -x "$f" ]; then
            venv_cmake="$f"
            break
        fi
    done
    if [ -n "$venv_cmake" ]; then
        local cmake_dir
        cmake_dir=$(dirname "$venv_cmake")
        export PATH="$cmake_dir:$PATH"
        return 0
    fi

    # 用 venv pip 安装
    if [ -f "${ASC_PROJECT}/.venv/bin/pip" ]; then
        echo "  安装 cmake 到 venv..."
        "${ASC_PROJECT}/.venv/bin/pip" install cmake --quiet
        export PATH="${ASC_PROJECT}/.venv/bin:$PATH"
    fi

    command -v cmake &>/dev/null
}

detect_os() {
    local uname_out
    uname_out=$(uname -s)
    case "$uname_out" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

# ============================================================
# Step 0: GPU 自动检测
# ============================================================

OS_TYPE=$(detect_os)

echo "=========================================="
echo " Asc 服务器环境初始化（自动检测 GPU）"
echo " 操作系统: ${OS_TYPE}"
echo "=========================================="
echo ""

echo "[检测] 扫描 GPU 硬件..."

GPU_TYPE="cpu"           # cpu | nvidia | amd | apple
GPU_INFO=""
GPU_COUNT=0

# --- Apple Metal 检测 (macOS) ---
if [ "$OS_TYPE" == "macos" ]; then
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
    # Apple Silicon 检测
    if sysctl -n hw.optional.arm64 2>/dev/null | grep -q "1" || \
       [ "$(uname -m)" == "arm64" ]; then
        GPU_TYPE="apple"
        GPU_COUNT=1
        # 获取芯片型号
        CHIP_NAME=$(system_profiler SPHardwareDataType 2>/dev/null | grep "Chip" | awk -F': ' '{print $2}' || echo "Apple Silicon")
        GPU_INFO="${CHIP_NAME}"
        echo "  [Apple Metal] 检测到 Apple Silicon"
        echo "    芯片: ${CHIP_NAME}"
    else
        # Intel Mac，可能有 AMD GPU
        GPU_TYPE="cpu"
        echo "  [macOS Intel] 未检测到 Apple Silicon，将使用 CPU 模式"
        echo "    CPU: ${CHIP}"
    fi
fi

# --- NVIDIA 检测 (Linux) ---
if [ "$GPU_TYPE" == "cpu" ] && [ "$OS_TYPE" == "linux" ]; then
    if command -v nvidia-smi &>/dev/null; then
        NVIDIA_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
        if [ -n "$NVIDIA_INFO" ]; then
            GPU_TYPE="nvidia"
            GPU_COUNT=0
            echo "  [NVIDIA] 检测到 GPU:"
            while IFS=',' read -r name vram; do
                name=$(echo "$name" | xargs)
                vram=$(echo "$vram" | xargs)
                GPU_COUNT=$((GPU_COUNT + 1))
                echo "    - ${name} (${vram})"
            done <<< "$NVIDIA_INFO"
            DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
            GPU_INFO="NVIDIA ${GPU_COUNT}x GPU (驱动 ${DRIVER_VER})"
        fi
    fi
fi

# --- AMD 检测 (Linux) ---
if [ "$GPU_TYPE" == "cpu" ] && [ "$OS_TYPE" == "linux" ]; then
    # 方法1: rocm-smi
    if command -v rocm-smi &>/dev/null; then
        AMD_INFO=$(rocm-smi --showproductname 2>/dev/null | grep -E "GPU\[" | head -8 || true)
        if [ -n "$AMD_INFO" ]; then
            GPU_TYPE="amd"
            GPU_COUNT=$(echo "$AMD_INFO" | wc -l)
            echo "  [AMD ROCm] 检测到 ${GPU_COUNT} 块 GPU"
            echo "$AMD_INFO" | while read -r line; do echo "    - $line"; done
            GPU_INFO="AMD ROCm ${GPU_COUNT}x GPU"
        fi
    fi

    # 方法2: /sys/class/drm
    if [ "$GPU_TYPE" == "cpu" ] && [ -d /sys/class/drm ]; then
        for card in /sys/class/drm/card*/device/vendor; do
            if [ -f "$card" ]; then
                vendor=$(cat "$card" 2>/dev/null || true)
                if [ "$vendor" == "0x1002" ]; then
                    GPU_TYPE="amd"
                    GPU_COUNT=$((GPU_COUNT + 1))
                fi
            fi
        done
        if [ "$GPU_TYPE" == "amd" ]; then
            echo "  [AMD] 检测到 ${GPU_COUNT} 块 GPU (通过 sysfs)"
            GPU_INFO="AMD ${GPU_COUNT}x GPU"
        fi
    fi

    # 方法3: lspci
    if [ "$GPU_TYPE" == "cpu" ] && command -v lspci &>/dev/null; then
        AMD_CARDS=$(lspci 2>/dev/null | grep -iE "VGA|3D|Display" | grep -i "AMD\|ATI\|Radeon" || true)
        if [ -n "$AMD_CARDS" ]; then
            GPU_TYPE="amd"
            GPU_COUNT=$(echo "$AMD_CARDS" | wc -l)
            echo "  [AMD] 检测到 ${GPU_COUNT} 块 GPU (通过 lspci)"
            GPU_INFO="AMD ${GPU_COUNT}x GPU"
        fi
    fi
fi

# --- CPU-only ---
if [ "$GPU_TYPE" == "cpu" ]; then
    CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | head -1 | cut -d: -f2 | xargs || \
                sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    echo "  [CPU] 未检测到独立 GPU，将使用 CPU 模式"
    echo "    CPU: ${CPU_MODEL}"
    echo "    核心数: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo '?')"
fi

echo ""
echo "=========================================="
echo " 编译方案: ${GPU_TYPE^^}"
echo "=========================================="
echo ""

# ============================================================
# Step 1: cmake
# ============================================================

echo "[Step 1/6] 检查 cmake..."
if ! ensure_cmake; then
    echo "  [错误] cmake 安装失败，无法继续"
    exit 1
fi
echo "  cmake: $(cmake --version | head -1)"

# ============================================================
# Step 2: 安装 GPU Toolkit
# ============================================================

echo ""
echo "[Step 2/6] 配置 GPU 工具链..."

HAS_GPU_ACCEL=0
CMAKE_EXTRA_ARGS=""

if [ "$GPU_TYPE" == "apple" ]; then
    # --- Apple Metal ---
    echo "  Apple Metal 无需额外安装，llama.cpp 内置支持"
    HAS_GPU_ACCEL=1
    CMAKE_EXTRA_ARGS="-DGGML_METAL=ON"

elif [ "$GPU_TYPE" == "nvidia" ]; then
    # --- NVIDIA CUDA ---
    if command -v nvcc &>/dev/null; then
        echo "  CUDA Toolkit 已安装: $(nvcc --version | tail -1)"
        HAS_GPU_ACCEL=1
    elif [ -f "${CUDA_INSTALL_DIR}/bin/nvcc" ]; then
        echo "  发现用户级 CUDA: ${CUDA_INSTALL_DIR}"
        export PATH="${CUDA_INSTALL_DIR}/bin:$PATH"
        export LD_LIBRARY_PATH="${CUDA_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
        HAS_GPU_ACCEL=1
    else
        echo "  需要安装 CUDA Toolkit 才能启用 GPU 加速"
        if prompt_yesno "  是否下载并安装 CUDA Toolkit 12.6 到 ${CUDA_INSTALL_DIR}？（约 4GB）"; then
            CUDA_RUNFILE="cuda_12.6.3_560.35.05_linux.run"
            CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/${CUDA_RUNFILE}"

            if [ ! -f "/tmp/${CUDA_RUNFILE}" ]; then
                echo "  下载中（约 4GB，可能需要几分钟）..."
                wget --show-progress -O "/tmp/${CUDA_RUNFILE}" "${CUDA_URL}" || {
                    echo "  [警告] CUDA Toolkit 下载失败"
                    rm -f "/tmp/${CUDA_RUNFILE}"
                }
            else
                echo "  发现已下载的 runfile，验证完整性..."
                # 检查文件大小（CUDA runfile 应 > 1GB）
                runfile_size=$(stat -c%s "/tmp/${CUDA_RUNFILE}" 2>/dev/null || echo "0")
                if [ "$runfile_size" -lt 1073741824 ]; then
                    echo "  runfile 不完整（${runfile_size} bytes），重新下载..."
                    rm -f "/tmp/${CUDA_RUNFILE}"
                    wget --show-progress -O "/tmp/${CUDA_RUNFILE}" "${CUDA_URL}" || {
                        echo "  [警告] CUDA Toolkit 下载失败"
                        rm -f "/tmp/${CUDA_RUNFILE}"
                    }
                fi
            fi

            if [ -f "/tmp/${CUDA_RUNFILE}" ]; then
                echo "  安装到 ${CUDA_INSTALL_DIR}..."
                sh "/tmp/${CUDA_RUNFILE}" --silent --toolkit --toolkitpath="${CUDA_INSTALL_DIR}" \
                    --no-opengl-libs --override 2>/dev/null || {
                    echo "  [警告] CUDA Toolkit 安装失败"
                }

                if [ -f "${CUDA_INSTALL_DIR}/bin/nvcc" ]; then
                    export PATH="${CUDA_INSTALL_DIR}/bin:$PATH"
                    export LD_LIBRARY_PATH="${CUDA_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
                    HAS_GPU_ACCEL=1
                    echo "  CUDA Toolkit 安装成功"
                fi
            fi
        else
            echo "  跳过 CUDA 安装，将编译 CPU-only 版本"
        fi
    fi

    if [ "$HAS_GPU_ACCEL" -eq 1 ]; then
        CMAKE_EXTRA_ARGS="-DGGML_CUDA=ON"
    fi

elif [ "$GPU_TYPE" == "amd" ]; then
    # --- AMD ROCm ---
    if command -v hipcc &>/dev/null; then
        echo "  ROCm 已安装: $(hipcc --version 2>/dev/null | head -1 || echo 'detected')"
        HAS_GPU_ACCEL=1
    elif [ -f "${ROCM_INSTALL_DIR}/bin/hipcc" ]; then
        echo "  发现用户级 ROCm: ${ROCM_INSTALL_DIR}"
        export PATH="${ROCM_INSTALL_DIR}/bin:$PATH"
        export LD_LIBRARY_PATH="${ROCM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
        HAS_GPU_ACCEL=1
    else
        echo "  需要安装 ROCm 才能启用 GPU 加速"
        echo "  ROCm 安装通常需要 sudo 权限，建议参考："
        echo "    https://rocm.docs.amd.com/projects/install-on-linux/docs/latest/install-methods.html"
        if prompt_yesno "  是否尝试下载 ROCm 最小安装包？（可能需要 sudo）" "N"; then
            ROCM_VER="6.3.1"
            echo "  尝试添加 ROCm apt 源..."
            wget -q "https://repo.radeon.com/amdgpu-install/${ROCM_VER}/ubuntu/24.04/amdgpu-install_${ROCM_VER}.60003-1_all.deb" \
                -O /tmp/amdgpu-install.deb 2>/dev/null && \
            sudo dpkg -i /tmp/amdgpu-install.deb 2>/dev/null && \
            sudo amdgpu-install --usecase=rocm --no-dkms -y 2>/dev/null && \
            HAS_GPU_ACCEL=1 || {
                echo "  [警告] ROCm 安装失败，将编译 CPU-only 版本"
            }
        else
            echo "  跳过 ROCm 安装，将编译 CPU-only 版本"
        fi
    fi

    if [ "$HAS_GPU_ACCEL" -eq 1 ]; then
        CMAKE_EXTRA_ARGS="-DGGML_HIPBLAS=ON"
        if command -v rocm-smi &>/dev/null; then
            AMD_ARCH=$(rocm-smi --showgpuid 2>/dev/null | grep "GPU arch" | head -1 | awk '{print $NF}' || true)
            if [ -n "$AMD_ARCH" ]; then
                CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DAMDGPU_TARGETS=${AMD_ARCH}"
            fi
        fi
    fi
fi

if [ "$HAS_GPU_ACCEL" -eq 0 ]; then
    echo "  将编译 CPU-only 版本（无 GPU 加速）"
    echo "  提示：安装 CUDA/ROCm 后重新运行此脚本即可启用 GPU 加速"
fi

# ============================================================
# Step 3: 克隆并编译 llama.cpp
# ============================================================

echo ""
echo "[Step 3/6] 编译 llama.cpp..."
LLAMA_DIR="$HOME/build/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "  克隆 llama.cpp 仓库（浅克隆加速）..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
if [ "$LLAMA_CPP_VERSION" != "main" ]; then
    echo "  切换到版本 ${LLAMA_CPP_VERSION}..."
    git fetch --depth 1 origin "${LLAMA_CPP_VERSION}" 2>/dev/null || true
    git checkout "${LLAMA_CPP_VERSION}" 2>/dev/null || echo "  使用 main 分支"
fi

# 清理旧构建
rm -rf build

echo "  配置 CMake..."
CMAKE_CMD="cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
if [ -n "$CMAKE_EXTRA_ARGS" ]; then
    CMAKE_CMD="${CMAKE_CMD} ${CMAKE_EXTRA_ARGS}"
fi
echo "  $CMAKE_CMD"
eval "$CMAKE_CMD"

echo "  编译中 (使用 $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) 线程)..."
cmake --build build --config Release -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# 安装二进制
echo "  安装 llama-server..."
mkdir -p "${INSTALL_PREFIX}/bin"
for bin in llama-server llama-cli llama-gguf; do
    if [ -f "build/bin/${bin}" ]; then
        cp -v "build/bin/${bin}" "${INSTALL_PREFIX}/bin/"
    fi
done

if [ -x "${INSTALL_PREFIX}/bin/llama-server" ]; then
    echo "  llama-server 安装成功!"
else
    echo "  [错误] llama-server 未找到"
    find build -name "llama-server" -type f 2>/dev/null
    exit 1
fi

# ============================================================
# Step 4: 配置 PATH
# ============================================================

echo ""
echo "[Step 4/6] 配置环境变量..."

# 根据系统选择 shell 配置文件
if [ "$OS_TYPE" == "macos" ]; then
    PROFILE_FILE="$HOME/.zshrc"
else
    PROFILE_FILE="$HOME/.bashrc"
fi

if ! grep -q "${INSTALL_PREFIX}/bin" "$PROFILE_FILE" 2>/dev/null; then
    echo "" >> "$PROFILE_FILE"
    echo "# Asc: llama.cpp" >> "$PROFILE_FILE"
    echo "export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\"" >> "$PROFILE_FILE"
    if [ "$GPU_TYPE" == "nvidia" ] && [ "$HAS_GPU_ACCEL" -eq 1 ]; then
        echo "export LD_LIBRARY_PATH=\"${CUDA_INSTALL_DIR}/lib64:\$LD_LIBRARY_PATH\"" >> "$PROFILE_FILE"
    elif [ "$GPU_TYPE" == "amd" ] && [ "$HAS_GPU_ACCEL" -eq 1 ]; then
        echo "export PATH=\"${ROCM_INSTALL_DIR}/bin:\$PATH\"" >> "$PROFILE_FILE"
        echo "export LD_LIBRARY_PATH=\"${ROCM_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH\"" >> "$PROFILE_FILE"
    fi
    echo "  已写入 ${PROFILE_FILE}"
fi
export PATH="${INSTALL_PREFIX}/bin:$PATH"

# ============================================================
# Step 5: 安装 Python 依赖
# ============================================================

echo ""
echo "[Step 5/6] 安装 Python 依赖..."
cd "${ASC_PROJECT}"

if [ -f ".venv/bin/pip" ]; then
    .venv/bin/pip install -q -e . 2>/dev/null
    .venv/bin/pip install -q -e ".[gui]" 2>/dev/null || echo "  [提示] Flask 安装失败，GUI 不可用"

    # llama-cpp-python (备用推理后端)
    if [ "$HAS_GPU_ACCEL" -eq 1 ] && [ "$GPU_TYPE" == "nvidia" ]; then
        echo "  安装 llama-cpp-python (CUDA)..."
        CMAKE_ARGS="-DGGML_CUDA=on" .venv/bin/pip install llama-cpp-python 2>/dev/null || \
            echo "  [提示] llama-cpp-python CUDA 编译失败，不影响 llama-server"
    elif [ "$HAS_GPU_ACCEL" -eq 1 ] && [ "$GPU_TYPE" == "amd" ]; then
        echo "  安装 llama-cpp-python (ROCm)..."
        CMAKE_ARGS="-DGGML_HIPBLAS=on" .venv/bin/pip install llama-cpp-python 2>/dev/null || \
            echo "  [提示] llama-cpp-python ROCm 编译失败，不影响 llama-server"
    elif [ "$HAS_GPU_ACCEL" -eq 1 ] && [ "$GPU_TYPE" == "apple" ]; then
        echo "  安装 llama-cpp-python (Metal)..."
        CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install llama-cpp-python 2>/dev/null || \
            echo "  [提示] llama-cpp-python Metal 编译失败，不影响 llama-server"
    else
        .venv/bin/pip install llama-cpp-python 2>/dev/null || \
            echo "  [提示] llama-cpp-python 安装失败"
    fi
else
    echo "  [警告] 未找到 .venv，跳过 Python 依赖安装"
fi

# ============================================================
# Step 6: 验证
# ============================================================

echo ""
echo "[Step 6/6] 验证安装..."
echo ""

PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  [OK] $name"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

check "llama-server" "test -x ${INSTALL_PREFIX}/bin/llama-server"
check "cmake" "command -v cmake"
check "Python venv" "test -d ${ASC_PROJECT}/.venv"
check "Asc 包" "${ASC_PROJECT}/.venv/bin/python -c 'import asc'"

if [ "$GPU_TYPE" == "nvidia" ] && [ "$HAS_GPU_ACCEL" -eq 1 ]; then
    check "CUDA nvcc" "command -v nvcc || test -x ${CUDA_INSTALL_DIR}/bin/nvcc"
fi
if [ "$GPU_TYPE" == "amd" ] && [ "$HAS_GPU_ACCEL" -eq 1 ]; then
    check "ROCm hipcc" "command -v hipcc || test -x ${ROCM_INSTALL_DIR}/bin/hipcc"
fi
if [ "$GPU_TYPE" == "apple" ]; then
    check "Apple Metal" "sysctl -n hw.optional.arm64 2>/dev/null || test \"$(uname -m)\" = arm64"
fi

echo ""
echo "=========================================="
echo " 安装完成: ${PASS} 通过, ${FAIL} 失败"
echo " GPU 加速: ${GPU_TYPE^^} ($([ $HAS_GPU_ACCEL -eq 1 ] && echo '已启用' || echo '未启用'))"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo ""
    echo " 启动 Asc 集群："
    echo ""
    echo "   # Master"
    echo "   cd ${ASC_PROJECT} && \\"
    echo "     .venv/bin/python -m asc master --host 0.0.0.0 --port 52414 --api-port 8080"
    echo ""
    echo "   # Worker"
    echo "   cd ${ASC_PROJECT} && \\"
    echo "     .venv/bin/python -m asc start --port 53414"
    echo ""
fi
