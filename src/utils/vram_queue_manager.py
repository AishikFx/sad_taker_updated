import threading
import queue
import time
import gc
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np


class ProcessStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessTask:
    """A single processing task with VRAM requirements"""
    task_id: str
    data: any
    vram_required_gb: float
    priority: int = 0  # Higher number = higher priority
    callback: Optional[Callable] = None
    status: ProcessStatus = ProcessStatus.QUEUED
    result: any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class VRAMMonitor:
    """Real-time VRAM monitoring and management"""
    
    def __init__(self, safety_margin: float = 0.9):
        self.safety_margin = safety_margin
        self.last_check = 0
        self.check_interval = 0.1  # Check every 100ms
        self.vram_history = []
        self.max_history = 50
        
    def get_vram_status(self) -> Dict[str, float]:
        """Get current VRAM status in GB"""
        current_time = time.time()
        
        if not torch.cuda.is_available():
            return {
                'total': 0.0,
                'allocated': 0.0,
                'free': 0.0,
                'available_safe': 0.0
            }
        
        # Cache results for short intervals to avoid overhead
        if current_time - self.last_check < self.check_interval:
            if hasattr(self, '_cached_status'):
                return self._cached_status
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated() / (1024**3)
        free = total - allocated
        available_safe = free * self.safety_margin
        
        status = {
            'total': total,
            'allocated': allocated,
            'free': free,
            'available_safe': available_safe
        }
        
        # Update history for trend analysis
        self.vram_history.append({
            'time': current_time,
            'free': free,
            'allocated': allocated
        })
        if len(self.vram_history) > self.max_history:
            self.vram_history.pop(0)
        
        self._cached_status = status
        self.last_check = current_time
        return status
    
    def can_allocate(self, required_gb: float) -> bool:
        """Check if we can safely allocate the required VRAM"""
        status = self.get_vram_status()
        return status['available_safe'] >= required_gb
    
    def force_cleanup(self):
        """Aggressive VRAM cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(0.1)  # Brief pause for cleanup
    
    def get_allocation_trend(self) -> str:
        """Analyze VRAM allocation trend"""
        if len(self.vram_history) < 5:
            return "stable"
        
        recent = self.vram_history[-5:]
        trend = sum(h['allocated'] for h in recent[-3:]) - sum(h['allocated'] for h in recent[:2])
        
        if trend > 0.5:  # More than 500MB increase
            return "increasing"
        elif trend < -0.5:  # More than 500MB decrease
            return "decreasing"
        else:
            return "stable"


class ParallelVRAMProcessor:
    """Dynamic parallel processor with VRAM-based task scheduling"""
    
    def __init__(self, max_queue_size: int = 1000, max_workers: int = 8):
        self.pending_queue = queue.Queue(maxsize=max_queue_size)  # Tasks waiting for VRAM
        self.active_tasks: Dict[str, ProcessTask] = {}  # Currently running tasks
        self.completed_tasks: List[ProcessTask] = []
        self.failed_tasks: List[ProcessTask] = []
        self.vram_monitor = VRAMMonitor()
        
        # Threading controls
        self.task_lock = threading.RLock()  # Reentrant lock
        self.scheduler_thread: Optional[threading.Thread] = None
        self.worker_threads: List[threading.Thread] = []
        self.max_workers = max_workers
        self.shutdown_flag = threading.Event()
        
        # VRAM tracking
        self.total_vram_allocated = 0.0  # Track our allocations
        self.vram_allocation_map: Dict[str, float] = {}  # task_id -> VRAM used
        
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'average_processing_time': 0.0,
            'vram_oom_count': 0,
            'parallel_tasks_peak': 0,
            'total_vram_allocated_peak': 0.0
        }
        
        # Start scheduler
        self.start_scheduler()
    
    def start_scheduler(self):
        """Start the dynamic VRAM scheduler"""
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            print("Dynamic VRAM scheduler started")
    
    def submit_task(self, task: ProcessTask) -> str:
        """Submit a task - will be scheduled based on VRAM availability"""
        try:
            self.pending_queue.put(task, timeout=1.0)
            print(f"Task {task.task_id} submitted to queue (VRAM req: {task.vram_required_gb:.1f}GB)")
            return task.task_id
        except queue.Full:
            raise RuntimeError("Task queue is full. Cannot accept more tasks.")
    
    def _scheduler_loop(self):
        """Main scheduler - dynamically assigns tasks based on VRAM"""
        print("Parallel VRAM scheduler started")
        
        while not self.shutdown_flag.is_set():
            try:
                # Check if we can start more tasks
                self._try_start_next_tasks()
                
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                # Brief pause to avoid CPU spinning
                time.sleep(0.05)  # 50ms check interval
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(0.1)
    
    def _try_start_next_tasks(self):
        """Try to start as many tasks as VRAM allows"""
        while not self.pending_queue.empty():
            try:
                # Get next task without removing from queue yet
                task = self.pending_queue.get_nowait()
                
                # Check if we have VRAM for this task
                if self._can_allocate_vram(task.vram_required_gb):
                    # Allocate VRAM and start task
                    self._allocate_vram(task)
                    self._start_task_worker(task)
                    print(f"Started task {task.task_id} (Active: {len(self.active_tasks)}, VRAM used: {self.total_vram_allocated:.1f}GB)")
                else:
                    # Put task back and wait
                    self.pending_queue.put(task)
                    break  # Can't start more tasks right now
                    
            except queue.Empty:
                break
    
    def _can_allocate_vram(self, required_gb: float) -> bool:
        """Check if we can allocate the required VRAM"""
        with self.task_lock:
            vram_status = self.vram_monitor.get_vram_status()
            available = vram_status['available_safe']
            
            # Account for our internal tracking
            effective_available = available - self.total_vram_allocated
            
            can_allocate = effective_available >= required_gb
            
            if not can_allocate:
                print(f"VRAM check: Need {required_gb:.1f}GB, Available: {effective_available:.1f}GB "
                      f"(System: {available:.1f}GB, Allocated by us: {self.total_vram_allocated:.1f}GB)")
            
            return can_allocate
    
    def _allocate_vram(self, task: ProcessTask):
        """Allocate VRAM for a task"""
        with self.task_lock:
            self.total_vram_allocated += task.vram_required_gb
            self.vram_allocation_map[task.task_id] = task.vram_required_gb
            
            # Update peak stats
            self.stats['total_vram_allocated_peak'] = max(
                self.stats['total_vram_allocated_peak'], 
                self.total_vram_allocated
            )
    
    def _free_vram(self, task: ProcessTask):
        """Free VRAM allocated by a task"""
        with self.task_lock:
            if task.task_id in self.vram_allocation_map:
                freed = self.vram_allocation_map.pop(task.task_id)
                self.total_vram_allocated = max(0, self.total_vram_allocated - freed)
                print(f"Freed {freed:.1f}GB VRAM from task {task.task_id} (Total allocated: {self.total_vram_allocated:.1f}GB)")
    
    def _start_task_worker(self, task: ProcessTask):
        """Start a worker thread for a specific task"""
        with self.task_lock:
            task.status = ProcessStatus.PROCESSING
            task.started_at = time.time()
            self.active_tasks[task.task_id] = task
            
            # Update peak parallel tasks
            self.stats['parallel_tasks_peak'] = max(
                self.stats['parallel_tasks_peak'], 
                len(self.active_tasks)
            )
        
        # Start worker thread
        worker = threading.Thread(
            target=self._process_task_worker, 
            args=(task,), 
            daemon=True,
            name=f"Worker-{task.task_id}"
        )
        worker.start()
        self.worker_threads.append(worker)
    
    def _process_task_worker(self, task: ProcessTask):
        """Worker function for processing a single task"""
        try:
            print(f"Worker processing task {task.task_id}")
            
            # Process the task
            if task.callback:
                task.result = task.callback(task.data)
            else:
                task.result = task.data  # Default: just return data
            
            task.status = ProcessStatus.COMPLETED
            task.completed_at = time.time()
            
            # Update statistics
            processing_time = task.completed_at - task.started_at
            with self.task_lock:
                self.stats['total_processed'] += 1
                
                # Update average processing time
                total_time = self.stats['average_processing_time'] * (self.stats['total_processed'] - 1)
                self.stats['average_processing_time'] = (total_time + processing_time) / self.stats['total_processed']
            
            print(f"Task {task.task_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            task.status = ProcessStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            with self.task_lock:
                self.stats['total_failed'] += 1
                
                if "out of memory" in str(e).lower():
                    self.stats['vram_oom_count'] += 1
                    print(f"OOM in task {task.task_id}. Total OOM count: {self.stats['vram_oom_count']}")
            
            print(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Always free VRAM and remove from active tasks
            self._free_vram(task)
            
            with self.task_lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
            
            # Force cleanup to ensure VRAM is actually freed
            self.vram_monitor.force_cleanup()
    
    def _cleanup_completed_tasks(self):
        """Move completed/failed tasks to appropriate lists"""
        with self.task_lock:
            completed_ids = []
            for task_id, task in self.active_tasks.items():
                if task.status in [ProcessStatus.COMPLETED, ProcessStatus.FAILED]:
                    completed_ids.append(task_id)
            
            for task_id in completed_ids:
                task = self.active_tasks.pop(task_id)
                if task.status == ProcessStatus.COMPLETED:
                    self.completed_tasks.append(task)
                else:
                    self.failed_tasks.append(task)
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all queued and active tasks to complete"""
        start_time = time.time()
        
        while True:
            with self.task_lock:
                pending_empty = self.pending_queue.empty()
                active_empty = len(self.active_tasks) == 0
            
            if pending_empty and active_empty:
                return True
            
            if timeout and (time.time() - start_time > timeout):
                return False
            
            time.sleep(0.1)
    
    def get_status(self) -> Dict:
        """Get current processor status"""
        with self.task_lock:
            vram_status = self.vram_monitor.get_vram_status()
            return {
                'pending_queue_size': self.pending_queue.qsize(),
                'active_tasks_count': len(self.active_tasks),
                'active_task_ids': list(self.active_tasks.keys()),
                'completed_count': len(self.completed_tasks),
                'failed_count': len(self.failed_tasks),
                'total_vram_allocated': self.total_vram_allocated,
                'vram_status': vram_status,
                'stats': self.stats.copy()
            }
    
    def force_cleanup_all(self):
        """Force cleanup of all VRAM and reset tracking"""
        with self.task_lock:
            print(f"Force cleanup: Clearing {self.total_vram_allocated:.1f}GB tracked allocation")
            self.total_vram_allocated = 0.0
            self.vram_allocation_map.clear()
        
        self.vram_monitor.force_cleanup()
        print("Force cleanup completed")
    
    def shutdown(self):
        """Shutdown the processor"""
        print("Shutting down parallel VRAM processor...")
        self.shutdown_flag.set()
        
        # Wait for scheduler
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        # Wait for active tasks to complete (with timeout)
        start_time = time.time()
        while len(self.active_tasks) > 0 and (time.time() - start_time < 10.0):
            print(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            time.sleep(0.5)
        
        # Force cleanup
        self.force_cleanup_all()
        
        print("Parallel VRAM processor shutdown complete")


# Global processor instance
_global_processor: Optional[ParallelVRAMProcessor] = None


def get_global_processor() -> ParallelVRAMProcessor:
    """Get or create the global parallel VRAM processor"""
    global _global_processor
    if _global_processor is None:
        _global_processor = ParallelVRAMProcessor()
    return _global_processor


def shutdown_global_processor():
    """Shutdown the global processor"""
    global _global_processor
    if _global_processor:
        _global_processor.shutdown()
        _global_processor = None