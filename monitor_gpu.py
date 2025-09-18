# GPU monitoring script - run this in a separate terminal while inference is running
import subprocess
import time

def monitor_gpu():
    """Monitor GPU usage during inference"""
    print("GPU Memory Usage Monitor")
    print("=" * 50)
    
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.stdout:
                memory_used, memory_total, gpu_util = result.stdout.strip().split(',')
                memory_used = int(memory_used.strip())
                memory_total = int(memory_total.strip())
                gpu_util = int(gpu_util.strip())
                
                memory_percent = (memory_used / memory_total) * 100
                
                print(f"GPU Utilization: {gpu_util}% | Memory: {memory_used}MB/{memory_total}MB ({memory_percent:.1f}%)")
                
                if memory_percent > 90:
                    print("⚠️  GPU memory is near capacity - consider reducing batch_size")
                elif memory_percent < 50:
                    print("💡 GPU memory usage is low - consider increasing batch_size")
                    
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    monitor_gpu()