import psutil
import time

class NetworkMonitor:
    def __init__(self, nic_name=None):
        self.nic_name = nic_name
        self.last_io = self._get_io()
        
    def _get_io(self):
        if self.nic_name:
            io_dict = psutil.net_io_counters(pernic=True)
            if self.nic_name in io_dict:
                return io_dict[self.nic_name]
        return psutil.net_io_counters()

    def get_diff(self):
        """获取从上次调用到现在产生的流量差值 (bytes)"""
        current_io = self._get_io()
        sent = current_io.bytes_sent - self.last_io.bytes_sent
        recv = current_io.bytes_recv - self.last_io.bytes_recv
        self.last_io = current_io
        return sent, recv

    @staticmethod
    def format_bytes(size):
        """格式化显示字节数"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"

def get_current_traffic():
    """获取当前网卡的总累计流量"""
    counters = psutil.net_io_counters()
    return counters.bytes_sent, counters.bytes_recv
