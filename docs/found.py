import psutil

# すべての親プロセスおよびその子プロセスをリストアップする関数
def list_all_processes_and_children():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            print(f"Parent process: {proc.info['name']} (PID: {proc.info['pid']})")
            parent = psutil.Process(proc.info['pid'])
            children = parent.children(recursive=True)
            for child in children:
                print(f"  Child process: {child.name()} (PID: {child.pid})")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

if __name__ == "__main__":
    list_all_processes_and_children()
