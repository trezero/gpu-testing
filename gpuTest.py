import subprocess
import time
from datetime import datetime


def measure_gpu_performance():
    # Get the date and time for filename
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"gpu_performance_log_{current_datetime}.txt"

    # Get initial GPU status
    init_gpu_status = subprocess.getoutput("nvidia-smi")

    # Extract GPU name
    gpu_name_line = [line for line in init_gpu_status.split("\n") if "NVIDIA" in line][0]
    gpu_name = gpu_name_line.split()[2]

    # Log the initial GPU status
    with open(log_filename, "w") as file:
        file.write(f"=== GPU: {gpu_name} ===\n\n")
        file.write("=== Initial GPU Status ===\n")
        file.write(init_gpu_status)
        file.write("\n\n")

    # Run the command and measure the time
    cmd = [
        "autotrain", "llm", "--train",
        "--data_path", ".",
        "--model", "mistralai/Mistral-7B-Instruct-v0.1",
        "--use_peft",
        "--target_modules", "q_proj,v_proj",
        "--use_int4",
        "--learning-rate", "2e-4",
        "--train_batch_size", "6",
        "--num_train_epochs", "3",
        "--trainer", "sft",
        "--project_name", "my-llm"
    ]

    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    with open(log_filename, "a") as file:
        for line in process.stdout:
            print(line, end='', flush=True)  # Print in real-time
            file.write(line)  # Log it

    stderr = process.stderr.read()
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Get post-execution GPU status
    post_gpu_status = subprocess.getoutput("nvidia-smi")

    # Log the command execution time and post-execution GPU status
    with open(log_filename, "a") as file:
        file.write(f"\n\nCommand Execution Time: {elapsed_time} seconds\n\n")
        file.write("=== Post Execution GPU Status ===\n")
        file.write(post_gpu_status)

    # If there's any error, log it
    if stderr:
        with open(log_filename, "a") as file:
            file.write("\n\n=== Errors ===\n")
            file.write(stderr)


if __name__ == "__main__":
    measure_gpu_performance()
