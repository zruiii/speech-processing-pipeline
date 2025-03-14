import requests
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading

class AudioDownloader:
    def __init__(self, max_workers=5, chunk_size=4*1024*1024):
        """
        初始化下载器
        
        Args:
            max_workers: 最大并发下载数
            chunk_size: 分块下载的大小(默认4MB)
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.session = self._create_session()
        self.pbar_lock = threading.Lock()
        
    def _create_session(self):
        """创建一个带有重试机制的会话"""
        session = requests.Session()
        
        # 配置重试策略
        retries = Retry(
            total=3,  # 总重试次数
            backoff_factor=0.5,  # 重试间隔
            status_forcelist=[500, 502, 503, 504]  # 需要重试的HTTP状态码
        )
        
        # 配置适配器
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=self.max_workers,  # 连接池大小
            pool_maxsize=self.max_workers  # 最大连接数
        )
        
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def download_file(self, url: str, save_path: str) -> bool:
        """下载单个文件"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 发起请求，设置超时
            response = self.session.get(url, stream=True, timeout=(5, 30))
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            
            with self.pbar_lock:  # 使用锁确保进度条正确显示
                progress = tqdm(
                    total=file_size,
                    unit='iB',
                    unit_scale=True,
                    desc=os.path.basename(save_path)
                )
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        with self.pbar_lock:
                            progress.update(size)
            
            progress.close()
            return True
            
        except Exception as e:
            print(f"下载出错 {save_path}: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False

    def download_batch(self, download_tasks):
        """并发下载多个文件"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for url, save_path in download_tasks:
                if not os.path.exists(save_path):
                    future = executor.submit(self.download_file, url, save_path)
                    futures.append((future, save_path))
            
            # 等待所有任务完成
            for future, save_path in futures:
                try:
                    success = future.result()
                    if success:
                        print(f"下载完成: {save_path}")
                    else:
                        print(f"下载失败: {save_path}")
                except Exception as e:
                    print(f"任务异常: {save_path}, {str(e)}")

# 使用示例
if __name__ == "__main__":
    MAX_NUM = 10                                 # 每个类别最多下载音频
    downloader = AudioDownloader(max_workers=5)  # 创建下载器实例
    
    download_tasks = []
    for file in os.listdir("data"):
        count = 0
        category = file.split("_")[0]
        save_dir = f"downloads/{category}"
        
        print(f"处理 {category} 类别播客")
        for sample in open(f"data/{file}"):
            episode_info = json.loads(sample.strip())
            media = episode_info['episode']['media']
            print(count)
            if "url" in media['source']:
                url = media['source']['url']
            elif "key" in media['source']:
                url = f"https://media.xyzcdn.net/{media['source']['key']}"
            eid = episode_info['episode']['eid']
            suffix = url.split(".")[-1]
            save_path = os.path.join(save_dir, f"{eid}.{suffix}")
            
            if not os.path.exists(save_path):
                download_tasks.append((url, save_path))
            
            count += 1
            if count == MAX_NUM:
                break
    
    # 执行批量下载
    downloader.download_batch(download_tasks)